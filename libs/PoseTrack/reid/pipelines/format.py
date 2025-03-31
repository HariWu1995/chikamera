
output_formats = [
    '%d',   # frame id
    '%d','%d','%d','%d','%.3e', # 4-point bbox & score
]

output_formats.extend(['%.10e'] * 2048)   # re-identification features

