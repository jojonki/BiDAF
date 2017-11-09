# for dot acess
# arg = {'name': 'jojonki', age: 100}
# conf = Config(**arg)
# print(conf.name) ==> 'jojonki'
class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
