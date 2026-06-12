def sum(*args):
    sum = 0
    for arg in args:
        sum += arg
    return sum


def dic(**kwargs):
    dic = {}
    for key, value in kwargs.items():
        dic[key] = value
    return dic


dic_data = {'model': 'deepseek-v4-pro',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello!'}
            ],
            'thinking': {'type': 'enabled'},
            'reasoning_effort': 'high',
            'stream': False
            }
for key in dic_data:
    print(f'{key} ---> {dic_data[key]}')
print("*" * 50)
for key in dic_data.keys():
    print(f'{key} ---> {dic_data[key]}')
print("*" * 50)
for key,value in dic_data.items():
    print(f'{key} ---> {value}')

list_data = [1,2,3,4,5]
print(*dic_data.values())
print(*list_data)