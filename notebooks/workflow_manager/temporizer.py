def temporizer(func,*args,**kargs):
    def just_in_time():
        return func(*args,**kargs)
    return just_in_time

def is_just_in_time(thing):
    #print(thing,"a",thing.__name__,callable(thing) and thing.__name__.split(".")[-1]=="just_in_time")
    return callable(thing) and thing.__name__.split(".")[-1]=="just_in_time"

def exec_just_in_time(thing):
    while is_just_in_time(thing):
        thing=thing()
    return thing
    