wget https://data.statmt.org/news-crawl/en/news.2023.en.shuffled.deduped.gz
gzip -d news.2023.en.shuffled.deduped.gz
# write random N lines to output
shuf -n 40000000 news.2023.en.shuffled.deduped -o monolingual.40M.en
