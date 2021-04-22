def checkUrl(url):
    words = ['/jobs', '/job', '.edu', 'profile']
    for word in words:
        index = None
        try:
            index = url.find(word)
        except:
            return False

        if index != -1:
            return False

    return True

def printUrl(url_list):
    for url in url_list:
        if checkUrl(url):
            print(url)

def countValidUrl(url_list):
    count = 0
    if checkUrl(url):
        count += 1
    return count

if __name__ == '__main__':
    url_list = ["https://www.politifact.com/factchecks/2020/apr/02/jerome-adams/hospitals-refute-surgeon-generals-claim-about-nurs/", "https://www.politifact.com/factchecks/2020/may/04/tucker-carlson/tucker-carlson-says-coronavirus-isnt-deadly-we-tho/"]
    printUrl(url_list)
    countValidUrl(url_list)