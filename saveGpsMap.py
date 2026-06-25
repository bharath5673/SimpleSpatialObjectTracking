import math
import os
import time
import folium
from PIL import Image, ImageDraw
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

GPS = [53.345520, -6.264284]
OUTPUT_HTML = "outputs/saveGpsMap.html"
OUTPUT_PNG = "outputs/saveGpsMap.png"

os.makedirs("outputs", exist_ok=True)

# m = folium.Map(location=GPS, zoom_start=19.5, tiles="CartoDB PositronNoLabels")
m = folium.Map(location=GPS, zoom_start=19.5, tiles="CartoDB Positron")
# folium.Marker(location=GPS).add_to(m)

m.save(OUTPUT_HTML)
print(f"Saved HTML: {OUTPUT_HTML}")

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=960,720")

driver = webdriver.Chrome(options=options)
driver.get(f"file://{os.path.abspath(OUTPUT_HTML)}")

time.sleep(3)
driver.save_screenshot(OUTPUT_PNG)
driver.quit()
print(f"Saved PNG: {OUTPUT_PNG}")

