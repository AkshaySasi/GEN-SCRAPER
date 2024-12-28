from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Function to get reviews from an Amazon product page using Selenium
def get_amazon_reviews(url):
    # Setup Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no UI)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Open the URL
    driver.get(url)
    
    # Wait for the reviews to load
    time.sleep(5)  # Adjust sleep time if necessary
    
    reviews = []
    review_elements = driver.find_elements(By.CSS_SELECTOR, "span[data-hook='review-body']")
    for review_element in review_elements:
        review_text = review_element.text.strip()
        reviews.append(review_text)
    
    driver.quit()
    return reviews

# Main function to scrape reviews and print them
def main(amazon_url):
    reviews = get_amazon_reviews(amazon_url)
    print(f"Total reviews found: {len(reviews)}")
    
    for i, review in enumerate(reviews, start=1):
        print(f"Review {i}: {review}\n")

if __name__ == "__main__":
    amazon_url = input("Enter the Amazon product URL: ")
    main(amazon_url)
