from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import  NoSuchElementException,ElementClickInterceptedException
import pandas as pd
from time import sleep
import os
import re 
from urllib.parse import urlparse
import sys

usernameIN = sys.argv[1]
passwordIN = sys.argv[2]
url = sys.argv[3]
print(f"Cào dữ liệu với tài khoản: {usernameIN}, URL: {url}")


options = Options()
options.add_argument("--start-maximized")  # Mở ở chế độ toàn màn hình
options.add_argument("--inprivate")  # Chế độ ẩn danh (InPrivate mode)
# options.add_argument("--headless")  # Chạy mà không mở trình duyệt
options.add_argument("--force-device-scale-factor=0.75") # Zoom màn hình lại 50%

driver = webdriver.Edge(options=options)
# url = "https://www.facebook.com/groups/ued.confessions?locale=vi_VN"
driver.get(url)


def find_element_with_fallback(driver, selectors):
    for selector in selectors:
        try:
            if selector['type'] == 'css':
                return driver.find_element(By.CSS_SELECTOR, selector['value'])
            elif selector['type'] == 'xpath':
                return driver.find_element(By.XPATH, selector['value'])
        except:
            continue
    raise NoSuchElementException(f"Không tìm thấy phần tử với các lựa chọn: {selectors}")

# Danh sách lựa chọn cho username
username_selectors = [
    {'type': 'xpath', 'value': '/html/body/div[1]/div/div[1]/div/div[5]/div/div/div[1]/div/div[2]/div/div/div/div[2]/form/div/div[3]/div/div/label/div/input'},
    {'type': 'xpath', 'value': '/html/body/div[1]/div/div[1]/div/div[2]/div[2]/div[2]/div/form/div[2]/div[1]/label/input'},
    {'type': 'xpath', 'value': '/html/body/div[1]/div[1]/div[1]/div/div[3]/div[2]/form/div[2]/div[1]/input'}
]

# Danh sách lựa chọn cho password
password_selectors = [
    {'type': 'xpath', 'value': '/html/body/div[1]/div/div[1]/div/div[5]/div/div/div[1]/div/div[2]/div/div/div/div[2]/form/div/div[4]/div/div/label/div/input'},
    {'type': 'xpath', 'value': '/html/body/div[1]/div/div[1]/div/div[2]/div[2]/div[2]/div/form/div[2]/div[2]/label/input'},
    {'type': 'xpath', 'value': '/html/body/div[1]/div[1]/div[1]/div/div[3]/div[2]/form/div[2]/div[2]/div/div/input'}
]

# Danh sách lựa chọn cho nút login
login_selectors = [
    {'type': 'css', 'value': 'div[aria-label="Accessible login button"]'},
    {'type': 'css', 'value': 'div[aria-label="Đăng nhập"]'},
    {'type': 'xpath', 'value': '/html/body/div[1]/div[1]/div[1]/div/div[3]/div[2]/form/div[2]/div[3]/button'}
]

# Tìm các phần tử
username = find_element_with_fallback(driver, username_selectors)
password = find_element_with_fallback(driver, password_selectors)
login = find_element_with_fallback(driver, login_selectors)

# Thực hiện đăng nhập

# usernameIN ="khoalolriot@gmail.com")
# passwordIN = "Akhoa@6204 "
username.send_keys(usernameIN)
password.send_keys(passwordIN)
login.click()
# print("Đăng nhập thành công")
# print("Đang cào.........")
def scroll():
    # Lấy chiều cao màn hình (viewport height) từ trình duyệt
    viewport_height = driver.execute_script("return window.innerHeight;")
    
    # Cuộn xuống một đoạn bằng chiều cao màn hình
    driver.execute_script(f"""
        window.scrollBy({{
            top: {viewport_height}, 
            behavior: 'smooth'
        }});
    """)
    sleep(2)  # Chờ 5 giây sau khi cuộn xuống
    
    # Cuộn lên lại một đoạn bằng chính chiều cao đã cuộn xuống
    driver.execute_script(f"""
        window.scrollBy({{
            top: {-viewport_height},
            behavior: 'smooth'
        }});
    """)
  # Chờ sau khi cuộn lên  
  
def scroll_to_post(post, time):
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", post)
    sleep(time)
    
def scroll_auto():    
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Cuộn xuống cuối trang
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(2)  # Chờ tải nội dung (nếu có)

        # Lấy chiều cao mới
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:  # Kiểm tra nếu đã cuộn đến cuối
            break
        last_height = new_height
    sleep(2)
    
parsed_url = urlparse(url)
path = parsed_url.path  # Lấy phần đường dẫn
result = re.sub( r'[^\w]','_',path.split('/')[-1]) 

path_normal = f'.scraped/{result}/post_facebook_{result}.csv'
path_reel = f'.scraped/{result}/post_reel_{result}.csv'


def saveOrCreate(result_post_normal, result_post_reel):
    # Tạo thư mục nếu chưa tồn tại
    dir_normal = os.path.dirname(path_normal)
    dir_reel = os.path.dirname(path_reel)

    # Kiểm tra và tạo thư mục nếu không tồn tại
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_reel, exist_ok=True)

    # Kiểm tra và tạo file CSV cho post_facebook nếu không tồn tại
    if not os.path.exists(path_normal):
        df_normal = pd.DataFrame(columns=['linkPoster', 'name', 'role', 'time', 'content', 'content_img', 'react_list', 'comment'])
        df_normal.to_csv(path_normal, index=True)

    # Nếu có dữ liệu mới, thêm vào file post_facebook.csv
    if result_post_normal:
        new_post_normal = pd.DataFrame(result_post_normal)
        df_normal = pd.read_csv(path_normal)
        df_normal = pd.concat([new_post_normal, df_normal], ignore_index=True)
        df_normal.to_csv(path_normal, index=True)

    # Kiểm tra và tạo file CSV cho post_reel nếu không tồn tại
    if not os.path.exists(path_reel):
        df_reel = pd.DataFrame(columns=['linkPoster', 'name', 'time', 'content', 'linkVideo', 'typeVideo', 'react_list', 'comment'])
        df_reel.to_csv(path_reel, index=True)

    # Nếu có dữ liệu mới, thêm vào file post_reel.csv
    if result_post_reel:
        new_post_reel = pd.DataFrame(result_post_reel)
        df_reel = pd.read_csv(path_reel)
        df_reel = pd.concat([new_post_reel, df_reel], ignore_index=True)
        df_reel.to_csv(path_reel, index=True)
        
def choose_new_post(driver):
    try: 
        chooseList = driver.find_element(By.CSS_SELECTOR, '[role="feed"] div:first-child')
        scroll_to_post(chooseList, 1)
        if any(option in chooseList.text for option in ['Phù hợp nhất', 'Hoạt động gần đây', 'Bài viết mới']):
            choice = chooseList.find_element(By.XPATH, './div/div')
            choice.click()
            sleep(1)
            banner_element = driver.find_element(By.XPATH, '//div[@role="banner"]')
            layer = banner_element.find_element(By.XPATH, 'following-sibling::div[1]')  
            choice_new = layer.find_element(By.XPATH, './div/div/div[2]/div/div/div[1]/div[1]/div/div/div/div/div/div/div/div[1]/div/div[3]')
            choice_new.click()
            sleep(5)
    except:
        pass
    sleep(2)
choose_new_post(driver)

def clickMoreContent(cmt):
    more_buttons = cmt.find_elements(By.CSS_SELECTOR, 'div[role = "button"]')
    for button in more_buttons:
        if button.text.strip() == 'Xem thêm':
            try:
                scroll_to_post(button, 0)
                button.click()
            except: 
                driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                button.click()
        sleep(1)
        
def clickMoreComment(cmt):
    more_buttons = cmt.find_elements(By.CSS_SELECTOR, 'div[role = "button"]')
    check_moreCmt = False
    for button in more_buttons:
        if 'Xem' in button.text and 'phản hồi' in button.text:
            try:
                scroll_to_post(button, 0)
                button.click()
            except: 
                driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                button.click()
            check_moreCmt = True
            break
        sleep(1)
    return check_moreCmt

def clickAllComment(cmt_box):
    more_button = cmt_box.find_element(By.XPATH, './div[3]')
    if more_button.text.strip():
        if 'Xem thêm' in more_button.text or 'Xem tất cả' in more_button.text:
            try:
                scroll_to_post(more_button, 0)
                more_button.click()
            except:
                driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                more_button.click()
        else:
            more_button = cmt_box.find_element(By.XPATH, './div[4]')
            if more_button.text.strip():
                if 'Xem thêm' in more_button.text or 'Xem tất cả' in more_button.text:
                    try:
                        scroll_to_post(more_button, 0)
                        more_button.click()
                    except:
                        driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                        more_button.click()


def close_popup_cmt(driver):
    try: 
        banner_element = driver.find_element(By.XPATH, '//div[@role="banner"]')
        layer = banner_element.find_element(By.XPATH, 'following-sibling::div[2]')  
        post = layer.find_element(By.CSS_SELECTOR, 'div[role="dialog"]')
        close = post.find_element(By.CSS_SELECTOR, 'div[aria-label="Đóng"]')
        close.click()
    except:
        pass
    sleep(1)


def extractInfo(info_box):
    try:
        user_link = info_box.find_element(By.XPATH, './div/div[2]/div/div[1]/span/div/h2/span/span/a').get_attribute('href')
    except:
        user_link = None
    user_name = info_box.find_element(By.XPATH, './div/div[2]/div/div[1]').find_element(By.CSS_SELECTOR, '.xt0psk2').text  
    time_post = info_box.find_element(By.XPATH, './div/div[2]/div/div[2]')

    try: 
        role = time_post.find_element(By.CSS_SELECTOR,'.x3nfvp2.x1kgmq87').text
    except NoSuchElementException:
        role = None

    try: 
        time_link = time_post.find_element(By.CSS_SELECTOR, 'use').get_attribute('xlink:href')
        time_post = driver.find_element(By.CSS_SELECTOR, f'text{time_link}').get_attribute('textContent')
    except NoSuchElementException:  
        time_post = time_post.text.replace('·','').strip()
    return user_link, user_name, time_post, role


def extractContent(content_box):
    # Lấy thông tin bài đăng
    try: # Có nội dung
        main_content = content_box.find_element(By.XPATH, './div[1]')
        try: 
            clickMoreContent(main_content)
        except NoSuchElementException: 
            pass
        main_content = content_box.find_element(By.XPATH, './div[1]').text
    except NoSuchElementException: # Không có nội dung
        main_content = None

    # Lấy ảnh trong bài đăng
    content_img_post = {}        
    try: # Có ảnh  
        content_img = content_box.find_element(By.XPATH, './div[2]')
        content_img_list = content_img.find_elements(By.CSS_SELECTOR, 'img')
        for i, img in enumerate(content_img_list):
            content_img_post[i] = img.get_attribute('src')
    except NoSuchElementException: # Không có ảnh
        pass
    return main_content, content_img_post

def extractReact(react_cmt_box):
    react_list = {
        "Thích" : 0,
        'Yêu thích' : 0,
        'Thương thương': 0,
        'Haha': 0,
        'Wow': 0,
        'Buồn': 0,
        'Phẫn nộ': 0
    }
    try: # Bài đăng có người thả cảm xúc
        react = react_cmt_box.find_element(By.XPATH,'./div/div/div[1]')

        react_icon = react.find_elements(By.XPATH,'./div/div[1]/div/div[1]/span/span/span')
        for react in react_icon: 
            rs = react.find_element(By.CSS_SELECTOR,'div[aria-label]').get_attribute('aria-label').split(':')[0].strip()
            for i in react_list: 
                if i== rs:
                    react_list[i] = react.find_element(By.CSS_SELECTOR,'div[aria-label]').get_attribute('aria-label').split(':')[1].strip().split()[0]
    except NoSuchElementException: # Bài đăng không có người thả cảm xúc
        pass
    
    return react_list
    
    
def extractComment_popUp(post, post_data_cmt):
    cmt_box = post.find_element(By.XPATH, './div/div/div/div[2]/div[2]/div/div/div[2]/div/div[2]/div[3]')
    scroll_to_post(cmt_box,0)
    cmt_list = cmt_box.find_elements(By.XPATH, './div[position() < last()]')
    for cmt in cmt_list:
        # Nhấn tất cả các nút xem thêm của comment đó
        while True:
            try:
                more_cmt_button_list = cmt.find_elements(By.CSS_SELECTOR, 'div[role="button"]')
                has_clicked = False 
                for button in more_cmt_button_list:
                    if "Xem" in button.text and "phản hồi" in button.text:
                        try:    
                            scroll_to_post(button, 0)
                            button.click()
                        except: 
                            driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                            button.click()
                        has_clicked = True
                        break  
                if not has_clicked:
                    break
                sleep(2)
            except Exception:
                break
        results = []
        content = cmt.text.split('\n')
        # Tiền xử lí comment
        try:
            i = 0
            for index_, line in enumerate(content):  
                if line == 'Thích': 
                    cmt_sub = content[i: index_ - 1]
                    results.append(cmt_sub)
                if line == "Chia sẻ" and content[index_ + 1] != 'Đã chỉnh sửa':
                    i = index_ + 1

                if line == 'Đã chỉnh sửa' and content[index_ + 1].isdigit():
                    i = index_ + 2
                elif line == 'Đã chỉnh sửa' and re.search(r'[a-zA-Z]', content[index_ + 1]):
                    i = index_ + 1 
                if line == 'Viết phản hồi công khai…':
                    i = index_ + 1        
        except: pass  
        # Lưu người viết và nội dung comment 
        for result in results: 
            for index_, line in enumerate(result):
                if index_ == 0: 
                    if line.isdigit():
                        name = result[index_ + 1]
                    else: 
                        name = result[index_]
                        num = index_
                content = ' '.join(result[num + 1: ])
                if line == '  ·' and result[index_ + 1] == 'Theo dõi':
                    content = ' '.join(result[index_ + 2 :])
                    break
                
            comment = {
                'name' : name, 
                'content': content
            }
            post_data_cmt.append(comment)
    return False

def extractComment(react_cmt_box):
    cmt_box = react_cmt_box.find_element(By.XPATH, './div/div')
    cmt_list = cmt_box.find_elements(By.XPATH,'./div')
    post_data_cmt = []
    scroll_to_post(react_cmt_box, 0)
    if len(cmt_list) < 4: # Không có comment nào trong bài đăng
        return None  
    
    try:
        more_button = cmt_box.find_element(By.XPATH, './div[3]')
        if 'Xem thêm' in more_button.text or 'Xem tất cả' in more_button.text:
            if more_button.text.strip():  # Kiểm tra xem nội dung có rỗng không
                try:
                    scroll_to_post(more_button, 0)
                    more_button.click()
                except:
                    driver.execute_script("window.scrollBy({top: -200, behavior: 'smooth'});")
                    more_button.click()
            else:
                pass
        else:
            try:
                more_button = cmt_box.find_element(By.XPATH, './div[4]')
                if more_button.text.strip():  # Kiểm tra nếu nội dung không rỗng
                    scroll_to_post(more_button, 0)
                    more_button.click()
                else:
                    pass 
            except Exception as e:
                pass 
    except:
        pass
    
    sleep(2)

    while True:
        try:                          
            banner_element = driver.find_element(By.XPATH, '//div[@role="banner"]')
            layer = banner_element.find_element(By.XPATH, 'following-sibling::div[2]')                                        
            post = layer.find_element(By.CSS_SELECTOR, 'div[role="dialog"]')
            check_moreCmt = extractComment_popUp(post, post_data_cmt)
        except:
            cmt_list = react_cmt_box.find_elements(By.XPATH, './div/div/div[position() > 2 and position() < last()]')
            for cmt in cmt_list:          
                try:  
                    clickMoreContent(cmt)
                    check_moreCmt = clickMoreComment(cmt)
                except: pass
            if not check_moreCmt: 
                for cmt in cmt_list:  
                    name = cmt.find_element(By.XPATH,'./div/div/div[2]/div[1]/div[1]/div/div/span').text
                    content = cmt.find_element(By.XPATH,'./div/div/div[2]/div[1]/div[1]/div/div/div').text
                    comment = {
                        'name' : name, 
                        'content': content
                    }
                    post_data_cmt.append(comment)
        if check_moreCmt:   
            continue 
        else: 
            break 
    return post_data_cmt


def refresh():
    driver.refresh()
    sleep(2)
    
    
# result_post_normal = []
# result_post_reel = []

# Tạo csv
saveOrCreate(None, None)
# Đọc file csv
prev_result_normal = pd.read_csv(path_normal)
prev_result_reel = pd.read_csv(path_reel)
print("The current url is: "+str(driver.current_url))
# Chưa crawl
if prev_result_normal.empty and  prev_result_reel.empty:
    print('This is the first time crawling this group')
else: # Đã crawl
    print('Crawled this group before')
index = 0
count_dupplicate = 0
post_list = driver.find_elements(By.CSS_SELECTOR, '[role="feed"] .x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
while True:
    # Lưu kết quả sau mỗi vòng lặp
    post_data_normal = {}
    post_data_reel = {} 
    check = False
    # Tắt cảnh báo
    try:
        close = driver.find_element(By.XPATH,'//*[@id="facebook"]/body/div[6]/div[1]/div/div[2]/div/div/div/div/div/div/div[1]/div/div[3]')
        close.click()
        sleep(5)
    except NoSuchElementException: 
        pass  
    
    # Scroll mỗi khi crawl hết trong danh sách có sẵn
    if index >= len(post_list):  
        scroll()
        sleep(4)
        post_list_new = driver.find_elements(By.CSS_SELECTOR, '[role="feed"] .x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
        if len(post_list_new) == len(post_list):
            check = True
            break 
        else:
            post_list = post_list_new 
        
    # Scroll to post
    scroll_to_post(post_list[index], 1)

    # Main
    try: # Post normal 
        print(f"Currently on post number: {index + 1}")
        print('This is a normal post.')
        
        # Lấy Xpath của bài post 
        main  = post_list[index].find_element(By.XPATH,'./div/div/div/div/div/div/div/div/div/div[13]/div/div')
        
        # Chia nhỏ Xpath thành 3 phần
        info_box  = main.find_element(By.XPATH,'./div[2]')            
        content_box = main.find_element(By.XPATH,'./div[3]')
        react_cmt_box = main.find_element(By.XPATH,'./div[4]')
        
        # Lấy thông tin người đăng
        user_link, user_name, time_post, role = extractInfo(info_box)
        
        # Lấy thông tin bài đăng
        main_content, content_img_post = extractContent(content_box)
        
        # Lấy thông tin lượt người thả cảm xúc
        react_list = extractReact(react_cmt_box)
        
        # Lấy thông tin comment của bài đăng
        post_data_cmt = extractComment(react_cmt_box)

        # Lưu thông tin toàn bộ bài post 
        post_data_normal = {
            'linkPoster': user_link if user_name != 'Người tham gia ẩn danh' else 'No link',
            'name': user_name,
            'role': role if role else 'Thành viên',
            'time': time_post,
            'content': main_content if main_content else 'No content',
            'content_img': content_img_post if content_img_post else 'No picture',
            'react_list': react_list,
            'comment': post_data_cmt if post_data_cmt else 'No comment'
        }
        index += 1
    except NoSuchElementException: # video
        print(f"Currently on post number {index + 1}:")
        print('This is a video post.')
        
        # Lấy ra Xpath cuả bài post
        main = post_list[index].find_element(By.XPATH,'./div/div/div/div/div/div/div/div/div/div[13]/div/div')
        content_box = main.find_element(By.XPATH,'./div[2]')
        react_cmt_box = main.find_element(By.XPATH,'./div[3]')  
        
        # Thông tin bài post
        link = content_box.find_element(By.CSS_SELECTOR, 'a')
        href = link.get_attribute('href')
        
        box = link.find_element(By.XPATH,'./div[1]/div[3]/div/div/div[1]/div/div/div/div/div/div[2]/span/span')
        
        linkPoster = box.find_element(By.CSS_SELECTOR,'a').get_attribute('href')
        
        box = box.text.split('·')
        typeVideo = box[0].replace('\n', '').strip()
        name = box[1].replace('\n', '').strip()
        time = box[2].replace('\n', '').strip()
        
        # Lấy ra nội dung bài post
        try: # Có nội dung
            main_content = link.find_element(By.XPATH,'./div[1]/div[2]/div/div/div[2]/span/div')
            try: 
                clickMoreContent(main_content)
            except NoSuchElementException: 
                pass
            main_content = link.find_element(By.XPATH, './div[1]/div[2]/div/div/div[2]/span/div').text
        except NoSuchElementException: # Không có nội dung
            main_content = None
            
        # Lấy thông tin lượt người thả cảm xúc
        react_list = extractReact(react_cmt_box)
        
        # Lấy thông tin comment của bài đăng
        post_data_cmt = extractComment(react_cmt_box)
        
        # Lưu thông tin bài post
        post_data_reel = {
            'linkPoster': linkPoster,
            'name': name,
            'time': time,
            'content': main_content if main_content else 'No content',
            'linkVideo': href,
            'typeVideo': typeVideo,
            'react_list': react_list,
            'comment': post_data_cmt if post_data_cmt else 'No comment'
        }
        
        index += 1

     # Đã crawl
    if not prev_result_normal.empty or not prev_result_reel.empty:
        if post_data_normal:  
            if 'role' in post_data_normal and post_data_normal['role'] != 'Quản trị viên':  
                if not prev_result_normal[
                    (prev_result_normal['content'] == post_data_normal['content']) &
                    (prev_result_normal['name'] == post_data_normal['name'])
                ].empty:
                    count_dupplicate += 1   
                    check = True
                    

        if post_data_reel: 
            if 'role' in post_data_reel and post_data_reel['role'] != 'Quản trị viên':  
                if not prev_result_reel[
                    (prev_result_reel['content'] == post_data_reel['content']) &
                    (prev_result_reel['linkVideo'] == post_data_reel['linkVideo'])
                ].empty:
                    count_dupplicate += 1
                    check = True 
    
    if count_dupplicate == 5: 
        break   

    if not check:
        # Thêm kết quả vào danh sách
        if post_data_normal:  
            # result_post_normal.append(post_data_normal)
            saveOrCreate([post_data_normal], None)
            print(post_data_normal)
        if post_data_reel:
            # result_post_reel.append(post_data_reel)
            saveOrCreate(None, [post_data_reel])
            print(post_data_reel)
        count_dupplicate = 0

    close_popup_cmt(driver)

print()
print('Crawl sucess')


driver.quit()

