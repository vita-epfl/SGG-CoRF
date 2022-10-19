try:
    from clearml import Task
    CLEARML_SET = True
except:
    CLEARML_SET = False
    pass

class ClearML_Singleton:
    __instance = None
    @staticmethod
    def getInstance():
        """ Static access method. """
        if not CLEARML_SET:
            return None
        if ClearML_Singleton.__instance == None:
            ClearML_Singleton()
        return ClearML_Singleton.__instance

    def __init__(self, task_name):
        """ Virtually private constructor. """
        if not CLEARML_SET:
            ClearML_Singleton.__instance = None
            return
        if ClearML_Singleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            ClearML_Singleton.__instance = Task.init(project_name='OpenPifPaf', task_name=task_name)
