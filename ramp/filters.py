
# class Filter(Storable):

#     def __init__(self, exclude_func=None, include_func=None):
#         self.exclude_func = exclude_func
#         self.include_func = include_func

#     def filter(self, df):
#         if self.include_func is not None:
#             df =

def filter_incomplete(df):
    df = df.dropna()
    return df
