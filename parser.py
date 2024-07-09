import yaml

with open('./config/config.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
     
# Print the values as a dictionary
print(data)