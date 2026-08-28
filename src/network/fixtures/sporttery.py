"""Official China Sports Lottery football fixture and fixed-bonus reader."""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


SPORTTERY_PAGE = 'https://www.sporttery.cn/jc/zqgdjj/'
SPORTTERY_MOBILE_PAGE = 'https://m.sporttery.cn/mjc/jsq/zqspf/'
CALCULATOR_API = (
    'https://webapi.sporttery.cn/gateway/uniform/football/'
    'getMatchCalculatorV1.qry?channel=c&poolCode=hhad,had,ttg,crs,hafu'
)
SELLING_API = (
    'https://webapi.sporttery.cn/gateway/jc/football/'
    'getSellingMatchListV1.qry?clientCode=3001'
)
BONUS_API = (
    'https://webapi.sporttery.cn/gateway/uniform/football/'
    'getFixedBonusV1.qry?clientCode=3001&matchId={match_id}'
)
RESULT_API = (
    'https://webapi.sporttery.cn/gateway/uniform/football/'
    'getUniformMatchResultV1.qry'
)

# Avoid retrying a known-denied legacy endpoint on every background pulse.
# The calculator feed remains active and this endpoint is retried after an hour.
_selling_api_suspended_until = 0.0
_SELLING_API_COOLDOWN_SECONDS = 60 * 60


class SportteryMobileClient:
    """Fast official mobile-calculator client (one request for all HAD matches)."""

    def __init__(self, timeout: float = 8.0, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
                'AppleWebKit/605.1.15 Mobile/15E148'
            ),
            'Origin': 'https://m.sporttery.cn',
            'Referer': SPORTTERY_MOBILE_PAGE,
            'Accept': 'application/json, text/plain, */*',
        })

    def _matches_from(self, url: str, value_key: Optional[str] = None) -> List[dict]:
        last_error = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if str(data.get('errorCode')) != '0':
                    raise RuntimeError(
                        f'{data.get("errorCode")} {data.get("errorMessage", "")}',
                    )
                value = data.get('value') or []
                if value_key is None:
                    return [dict(match) for match in value]
                groups = value.get(value_key) or []
                return [dict(match) for group in groups
                        for match in (group.get('subMatchList') or [])]
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(str(last_error))

    @staticmethod
    def _merge_match_feeds(*feeds: List[dict]) -> List[dict]:
        """Union official feeds while retaining calculator odds when present."""
        merged = {}
        anonymous = []
        for feed in feeds:
            for match in feed:
                match_id = str(match.get('matchId') or '')
                if not match_id:
                    anonymous.append(dict(match))
                    continue
                merged[match_id] = {**merged.get(match_id, {}), **dict(match)}
        return list(merged.values()) + anonymous

    def selling_matches(self) -> List[dict]:
        """Return the union of the full selling list and calculator markets.

        The calculator endpoint can contain only the subset offered in one or
        more pools.  It must therefore never be treated as the complete card.
        """
        global _selling_api_suspended_until
        feeds, errors = [], []
        sources = []
        if time.monotonic() >= _selling_api_suspended_until:
            sources.append((SELLING_API, None, '全量在售'))
        sources.append((CALCULATOR_API, 'matchInfoList', '计算器赔率'))
        for url, value_key, label in sources:
            try:
                feeds.append(self._matches_from(url, value_key))
            except RuntimeError as error:
                errors.append(f'{label}={error}')
                if url == SELLING_API and '403' in str(error):
                    _selling_api_suspended_until = (
                        time.monotonic() + _SELLING_API_COOLDOWN_SECONDS
                    )
        if not feeds:
            raise RuntimeError(f'竞彩网官方接口读取失败：{"; ".join(errors)}')
        if errors:
            logging.warning('竞彩官方数据源部分失败：%s', '；'.join(errors))
        return self._merge_match_feeds(*feeds)

    def fixed_bonus_history(self, match_id: str) -> dict:
        """Return official chronological fixed-bonus history for one match."""
        last_error = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    BONUS_API.format(match_id=str(match_id)),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if str(data.get('errorCode')) != '0':
                    raise RuntimeError(
                        f'{data.get("errorCode")} {data.get("errorMessage", "")}',
                    )
                return ((data.get('value') or {}).get('oddsHistory') or {})
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f'官方固定奖金历史读取失败：{last_error}')

    def snapshot(self, output: Path) -> List[dict]:
        matches = self.selling_matches()
        output.parent.mkdir(parents=True, exist_ok=True)
        # A feed may shrink after a sales cutoff or during an upstream partial
        # response.  Preserve every match already observed on today's card.
        if output.exists():
            try:
                cached = json.loads(output.read_text(encoding='utf-8'))
                matches = self._merge_match_feeds(
                    list(cached.get('matches') or []), matches,
                )
            except (OSError, ValueError, TypeError):
                logging.warning('无法合并今日竞彩原始快照：%s', output)
        output.write_text(json.dumps({
            'fetchedAt': datetime.now(timezone.utc).isoformat(),
            'source': SPORTTERY_MOBILE_PAGE,
            'matches': matches,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return matches


class SportteryResultClient:
    """Official settled football results used for leakage-free daily review."""

    def __init__(self, timeout: float = 8.0, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Origin': 'https://www.sporttery.cn',
            'Referer': 'https://www.sporttery.cn/jc/zqsgkj/',
            'Accept': 'application/json, text/plain, */*',
        })

    def settled_matches(self, begin_date: str, end_date: str) -> List[dict]:
        """Fetch every published result in an inclusive date range."""
        page, rows = 1, []
        while True:
            params = {
                'matchBeginDate': begin_date,
                'matchEndDate': end_date,
                'leagueId': '',
                'pageSize': 100,
                'pageNo': page,
                'isFix': 0,
                'matchPage': 1,
                'pcOrWap': 1,
            }
            last_error = None
            for attempt in range(self.retries):
                try:
                    response = self.session.get(
                        RESULT_API, params=params, timeout=self.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if str(data.get('errorCode')) != '0':
                        raise RuntimeError(
                            f'{data.get("errorCode")} {data.get("errorMessage", "")}',
                        )
                    value = data.get('value') or {}
                    rows.extend(dict(match) for match in value.get('matchResult') or [])
                    pages = int(value.get('pages') or 1)
                    break
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    last_error = error
                    if attempt + 1 < self.retries:
                        time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f'竞彩网官方赛果读取失败：{last_error}')
            if page >= pages:
                return rows
            page += 1


class SportteryScraper:
    """Read public Sporttery JSON through its browser page origin.

    The public API rejects many non-browser requests. Executing the same fetch
    used by the official page avoids brittle HTML parsing and preserves fields.
    """

    def __init__(self, headless: bool = True, timeout: float = 25.0):
        mac_chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        browser_path = (shutil.which('google-chrome')
                        or shutil.which('google-chrome-stable')
                        or (str(mac_chrome) if mac_chrome.exists() else None))
        driver_path = shutil.which('chromedriver')
        if browser_path is None:
            raise RuntimeError('未找到 Chrome，无法启用竞彩浏览器备用抓取。')
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless=new')
        options.add_argument('--lang=zh-CN')
        options.add_argument('--window-size=1440,1000')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.binary_location = browser_path
        # With no explicit driver Selenium Manager resolves a compatible
        # driver.  This is the normal macOS installation path.
        service = ChromeService(driver_path) if driver_path else ChromeService()
        self._driver = Chrome(service=service, options=options)
        self._driver.set_page_load_timeout(timeout)
        self._driver.set_script_timeout(timeout)
        self._timeout = timeout

    def __enter__(self):
        self._driver.get(SPORTTERY_PAGE)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        try:
            self._driver.quit()
        except Exception:
            pass

    def _fetch_json(self, url: str) -> dict:
        script = """
            const done = arguments[arguments.length - 1];
            fetch(arguments[0], {credentials: 'omit'})
              .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
              })
              .then(data => done({ok: true, data}))
              .catch(error => done({ok: false, error: String(error)}));
        """
        result = WebDriverWait(self._driver, self._timeout).until(
            lambda driver: driver.execute_async_script(script, url),
        )
        if not result or not result.get('ok'):
            raise RuntimeError(f'竞彩网接口读取失败：{(result or {}).get("error", "无返回")}.')
        data = result['data']
        if data.get('errorCode') != 0:
            raise RuntimeError(f'竞彩网接口错误：{data.get("errorCode")} {data.get("errorMessage", "")}')
        return data

    def selling_matches(self) -> List[dict]:
        try:
            data = self._fetch_json(SELLING_API)
            return list(data.get('value') or [])
        except RuntimeError:
            # The official page itself populates this select from the same API.
            WebDriverWait(self._driver, self._timeout).until(
                lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '#matchList option')) > 1,
            )
            rows = []
            for option in self._driver.find_elements(By.CSS_SELECTOR, '#matchList option'):
                match_id = option.get_attribute('value')
                if match_id and match_id.isdigit():
                    rows.append({'matchId': match_id, 'displayText': option.text.strip()})
            return rows

    def fixed_bonus(self, match_id: str) -> dict:
        try:
            return self._fetch_json(BONUS_API.format(match_id=match_id)).get('value') or {}
        except RuntimeError:
            self._driver.get(f'{SPORTTERY_PAGE}?m={match_id}')
            WebDriverWait(self._driver, self._timeout).until(
                lambda driver: driver.find_elements(By.CSS_SELECTOR, '#had_tb tr'),
            )
            cells = self._driver.find_elements(By.CSS_SELECTOR, '#had_tb tr:first-child td')
            values = [cell.text.strip() for cell in cells]
            if len(values) < 4 or '暂无数据' in ''.join(values):
                return {'oddsHistory': {'hadList': []}}
            return {'oddsHistory': {'hadList': [{
                'updateDate': values[0], 'h': values[1], 'd': values[2], 'a': values[3],
            }]}}

    def snapshot(
            self,
            output: Path,
            include_bonus: Optional[Callable[[dict], bool]] = None,
    ) -> List[dict]:
        matches = self.selling_matches()
        rows = []
        for match in matches:
            match_id = str(match.get('matchId', ''))
            item = dict(match)
            should_fetch = include_bonus(item) if include_bonus is not None else True
            item['fixedBonus'] = self.fixed_bonus(match_id) if match_id and should_fetch else {}
            rows.append(item)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            'fetchedAt': datetime.now(timezone.utc).isoformat(),
            'source': SPORTTERY_PAGE,
            'matches': rows,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        return rows


def latest_had_odds(fixed_bonus: dict) -> Optional[Dict[str, float]]:
    """Return latest non-handicap H/D/A fixed bonuses, if offered."""
    direct = fixed_bonus.get('had') or {}
    if direct:
        try:
            return {'H': float(direct['h']), 'D': float(direct['d']), 'A': float(direct['a'])}
        except (KeyError, TypeError, ValueError):
            pass
    history = fixed_bonus.get('oddsHistory') or {}
    rows = history.get('hadList') or []
    if not rows:
        return None
    row = rows[0]
    try:
        return {'H': float(row['h']), 'D': float(row['d']), 'A': float(row['a'])}
    except (KeyError, TypeError, ValueError):
        return None


def latest_hhad_odds(match: dict) -> Optional[Dict[str, float]]:
    """Return official handicap and H/D/A fixed bonuses, if offered."""
    handicap = match.get('hhad') or {}
    if not handicap:
        return None
    try:
        return {
            'line': float(handicap['goalLine']),
            'H': float(handicap['h']),
            'D': float(handicap['d']),
            'A': float(handicap['a']),
        }
    except (KeyError, TypeError, ValueError):
        return None
