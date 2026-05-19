from paraview.simple import *
from paraview import catalyst
from paraview import print_info


options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"

producer = TrivialProducer(registrationName="fides")


def catalyst_execute(info):
    producer.UpdatePipeline()
