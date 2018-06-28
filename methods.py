import logging

class Methods:
    """
    模型方法类
    """

    _methods = []
    _modules = {}
    _funcs = {}

    def __init__(self, name):
        self._name = name

    def parseFunc(self, func):
        func_info = func.split(".")
        if(len(func_info) != 2):
            return None, None
        try:
            _modules = __import__(func_info[0])
            if(not hasattr(_modules, func_info[1])):
                return None, None
            return func_info[0], func_info[1]
        except Exception:
            return None, None

    def register(self, method, func):
        if method not in self._methods:
            module, func_name = self.parseFunc(func)
            if module is None:
                logging.error("Not have the func: %s" % func)
                return False
            self._methods.append(method)
            self._modules[method] = module
            self._funcs[method] = func_name
        else:
            logging.info("The method exist!")

    def unregister(self, method, func):
        if method not in self._methods:
            return
        _methods.remove(method)

        del self._modules[method]
        del self._funcs[method]

    def has(self, method):
        return method in self._methods

    def exe(self, method, **args):
        if not self.has(method):
            logging.error("The method %s not exist" % method)
            return
        module = __import__(self._modules[method])
        func = getattr(module, self._funcs[method])
        func(args)

