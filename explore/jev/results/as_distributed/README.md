Judge outputs and intermediate files produced before the url column was repaired (scripts/a1_realign_urls.py).
Row numbers (a, b, row) refer to `load(repaired=False)`: the file as distributed, sorted by URL date and URL. For
rows after 6 June 2013 the headline that was judged belongs to another article than the features and shares of
that row, so accuracies computed from these files are not valid. They are kept only for the as-distributed versus
repaired comparison (scripts/a2_alignment_validation.py); `api_mash_titles_section.csv` carries url_a / url_b for that.
