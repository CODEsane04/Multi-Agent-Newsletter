from agents.scout import fetch_rss_feeds

result = fetch_rss_feeds()

count = 1
for ni in result :
    print(f"{count}. {ni['snippet']}")
    print("\n")
    count += 1

    if count > 5 : 
        break