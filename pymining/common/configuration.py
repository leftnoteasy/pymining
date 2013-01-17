from xml.dom import minidom

class Configuration:
    mCurNode = None

    def __init__(self, node):
        self.mCurNode = node

    
    """
    get first child
    """
    def GetChild(self, name):
        for node in self.mCurNode.childNodes:
            if node.nodeName == name:
                return Configuration(node)
        return None

    def GetChilds(self, name):
        nodes = []
        for node in self.mCurNode.childNodes:
            if node.nodeName == name:
                nodes.append(Configuration(node))
        return nodes

    def GetName(self):
        return self.mCurNode.nodeName

    def GetValue(self):
        return self.mCurNode.firstChild.data

    @staticmethod
    def FromFile(path):
        return Configuration(minidom.parse(path).childNodes[0])

"""
if __name__ == "__main__":
    cfg = Configuration.FromFile("sandbox/test.xml")
    print cfg.GetName()
    print cfg.GetValue()
    cfg1 = cfg.GetChild("hello")
    print cfg1.GetName()
    print cfg1.GetValue()
    cfgs = cfg.GetChilds("world")
    for c in cfgs:
        print c.GetName()
        print c.GetValue()
"""
