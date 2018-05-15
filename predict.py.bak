import yaml

# load config file
with open('param.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# create and run model
sModel = 'model.'+cfg['model']
model = __import__(sModel, globals(), locals(), ['run'], 0)
model.run(cfg)
