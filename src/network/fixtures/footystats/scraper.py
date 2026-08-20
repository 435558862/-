import json
import logging
import shutil
import time
from datetime import datetime
import pandas as pd
from typing import Optional
from lxml import html
from selenium.webdriver import Chrome, Firefox, Edge, ChromeOptions, FirefoxOptions, EdgeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from src.network.netutils import check_internet_connection


class FootyStatsScraper:
    """ FootyStats scraper, which opens FootyStats webpage via a web browser and parses the fixture table. """

    def __init__(self):
        self._page_load_timeout = 12.0
        self._poll_frequency = 0.5

        with open('storage/network/browser.json', mode='r') as jsonfile:
            browser = json.load(jsonfile)['application']

        if browser == 'chrome':
            browser_path = (
                shutil.which('google-chrome')
                or shutil.which('google-chrome-stable')
                or shutil.which('chromium')
                or shutil.which('chromium-browser')
            )
            if browser_path is None:
                raise RuntimeError('WSL 中未安装 Chrome/Chromium，无法获取在线赛程。')
            driver_path = shutil.which('chromedriver')
            if driver_path is None:
                raise RuntimeError('WSL 中未安装 Chromedriver，无法获取在线赛程。')
            options = ChromeOptions()
            options.add_argument('--incognito')
            options.add_argument('--lang=en-US')
            options.add_argument('--headless=new')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--window-size=1440,1200')
            options.binary_location = browser_path
            self._web_driver = Chrome(service=ChromeService(driver_path), options=options)
        elif browser == 'firefox':
            browser_path = shutil.which('firefox')
            if browser_path is None:
                raise RuntimeError('WSL 中未安装 Firefox，无法获取在线赛程。')
            driver_path = shutil.which('geckodriver')
            if driver_path is None:
                raise RuntimeError('WSL 中未安装 Geckodriver，无法获取在线赛程。')
            options = FirefoxOptions()
            options.add_argument('--incognito')
            options.add_argument('-headless')
            options.set_preference('intl.accept_languages', 'en-US, en')
            options.binary_location = browser_path
            self._web_driver = Firefox(service=FirefoxService(driver_path), options=options)
        elif browser == 'edge':
            browser_path = shutil.which('microsoft-edge') or shutil.which('microsoft-edge-stable')
            if browser_path is None:
                raise RuntimeError('WSL 中未安装 Microsoft Edge，无法获取在线赛程。')
            driver_path = shutil.which('msedgedriver')
            if driver_path is None:
                raise RuntimeError('WSL 中未安装 Edge WebDriver，无法获取在线赛程。')
            options = EdgeOptions()
            options.add_argument('--incognito')
            options.add_argument('--lang=en-US')
            options.add_argument('--headless=new')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.binary_location = browser_path
            self._web_driver = Edge(service=EdgeService(driver_path), options=options)
        else:
            raise NotImplementedError(
                f'Not Implemented browser: "{browser}". '
                f'Only Chrome, Firefox and Edge are currently supported.'
            )
        # Selenium's default page-load timeout is unbounded.  FootyStats may
        # keep analytics/advertising requests open, which used to leave the
        # fixtures dialog stuck forever even though the work runs in a thread.
        self._web_driver.set_page_load_timeout(self._page_load_timeout)
        self._web_driver.set_script_timeout(self._page_load_timeout)

    def load_page(self, fixture_url: str) -> bool:
        """ Loads the FootyStats webpage and waits until loading state is ready. """

        # Check internet connection first.
        if not check_internet_connection():
            return False

        # Load webpage using the web driver.
        try:
            self._web_driver.get(url=fixture_url)
        except Exception as error:
            logging.info(f'Fixture page load failed or timed out: {error}')
            return False

        try:
            WebDriverWait(self._web_driver, timeout=self._page_load_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.full-matches-table'))
            )
        except Exception as _:
            logging.info('Timed out waiting for fixture table to load.')
            return False

        time.sleep(1.0)
        return True

    def parse_fixture_table(self, date_str: str) -> Optional[pd.DataFrame]:
        """ Parses the fixture table. The fixture table should be displayed on the web page! """

        # Reads the odd from the provided span.
        def get_odd(span) -> str:
            text = span.text

            # If text is inside child element, it attempts to read the text inside the child element instead.
            # If no odd text is found, it returns 1.0, which id the default odd value.
            if text is None:
                for child in span:
                    text = child.text

                    if text is not None:
                        break

            return text.replace('\n', '').replace('\t', '') if text is not None else '1.0'

        tree = html.fromstring(self._web_driver.page_source)
        table_elements = tree.xpath('//div[contains(@class, "full-matches-table mt1e")]')

        if len(table_elements) == 0:
            raise RuntimeError('Could not find "full-matches-table mt1e" table class.')

        # Searching the requested table by date.
        # The site has used several header variants ("Aug 6 ~", "Aug 06",
        # and headers containing a weekday/year). Compare normalized dates
        # instead of requiring one exact piece of text.
        requested_date = None
        for fmt in ('%b %d', '%b %d %Y'):
            try:
                requested_date = datetime.strptime(date_str, fmt).strftime('%b %d').lower()
                break
            except ValueError:
                pass
        requested_date = requested_date or date_str.strip().lower()
        requested_table = None
        for table in table_elements:
            date_element = table.find('h2')

            if date_element is None:
                continue

            header = ' '.join(date_element.text_content().split()).lower()
            normalized_header = header.replace(' 0', ' ')
            normalized_request = requested_date.replace(' 0', ' ')
            if normalized_request in normalized_header:
                requested_table = table
                break

        if requested_table is None:
            logging.info(f'Could not find the selected date: "{date_str}" in a table header.')
            return None

        # Parsing fixture table.
        home_teams = []
        away_teams = []
        odds_1 = []
        odds_x = []
        odds_2 = []
        for ul in requested_table.findall('.//ul')[1:]:
            # Parsing teams.
            home_teams.append(ul.findall('.//a')[0].find('.//span').text)
            away_teams.append(ul.findall('.//a')[2].find('.//span').text)

            # Parsing odds.
            odd_spans = ul.findall('li')[-1].xpath('.//span[contains(@class, "hover-modal-parent")]')
            odd_1 = get_odd(span=odd_spans[0])
            odds_1.append(odd_1)
            odd_x = get_odd(span=odd_spans[1])
            odds_x.append(odd_x)
            odd_2 = get_odd(span=odd_spans[2])
            odds_2.append(odd_2)

        # Add year to dates.
        df = pd.DataFrame({
            'Home': home_teams,
            'Away': away_teams,
            '1': odds_1,
            'X': odds_x,
            '2': odds_2
        })
        return df

    def quit(self):
        self._web_driver.quit()
