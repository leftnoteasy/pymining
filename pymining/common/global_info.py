#save materials produced during data-mining
class GlobalInfo:
    #dict store term -> id
    termToId = {} 
    #dict store id -> term
    idToTerm = {}
    #dict store term -> how-many-docs-have-term
    idToDocCount = {}
    #dict store class -> how-many-docs-contained
    classToDocCount = {}
    #inverse document frequent of termId
    idToIdf = {}
    #transfered id to termId
    newIdToId = {}

    #filename of above
    nameTermToId = ""
    nameIdToTerm = ""
    nameIdToDocCount = ""
    nameClassToDocCount = ""
    nameIdToIdf = ""
    nameNewIdToId = ""

    #isInit
    isInit = False

    #curNode
    curNode = None

    @staticmethod
    def Init(config, nodeName, loadFromFile = False):
        GlobalInfo.termToId = {}
        GlobalInfo.idToTerm = {}
        GlobalInfo.idToDocCount = {}
        GlobalInfo.classToDocCount = {}
        GlobalInfo.idToIdf = {}

        GlobalInfo.curNode = config.GetChild(nodeName)
        GlobalInfo.nameTermToId = GlobalInfo.curNode.GetChild("term_to_id").GetValue()
        GlobalInfo.nameIdToTerm = GlobalInfo.curNode.GetChild("id_to_term").GetValue()
        GlobalInfo.nameIdToDocCount = GlobalInfo.curNode.GetChild("id_to_doc_count").GetValue()
        GlobalInfo.nameClassToDocCount = GlobalInfo.curNode.GetChild("class_to_doc_count").GetValue()
        GlobalInfo.nameIdToIdf = GlobalInfo.curNode.GetChild("id_to_idf").GetValue()
        GlobalInfo.nameNewIdToId = GlobalInfo.curNode.GetChild("newid_to_id").GetValue()

        if (loadFromFile):
            GlobalInfo.__ReadDict(GlobalInfo.termToId, GlobalInfo.nameTermToId, "str", "int")
            GlobalInfo.__ReadDict(GlobalInfo.idToTerm, GlobalInfo.nameIdToTerm, "int", "str")
            GlobalInfo.__ReadDict(GlobalInfo.idToDocCount, GlobalInfo.nameIdToDocCount, "int", "int")
            GlobalInfo.__ReadDict(GlobalInfo.classToDocCount, GlobalInfo.nameClassToDocCount, "int", "int")
            GlobalInfo.__ReadDict(GlobalInfo.idToIdf, GlobalInfo.nameIdToIdf, "int", "float")
            GlobalInfo.__ReadDict(GlobalInfo.newIdToId, GlobalInfo.nameNewIdToId, "int", "int")
        
        GlobalInfo.isInit = True

    @staticmethod
    def Write():
        if (not GlobalInfo.isInit):
            print "call init before write()"
            return False
        GlobalInfo.__WriteDict(GlobalInfo.termToId, GlobalInfo.nameTermToId)
        GlobalInfo.__WriteDict(GlobalInfo.idToTerm, GlobalInfo.nameIdToTerm)
        GlobalInfo.__WriteDict(GlobalInfo.idToDocCount, GlobalInfo.nameIdToDocCount)
        GlobalInfo.__WriteDict(GlobalInfo.classToDocCount, GlobalInfo.nameClassToDocCount)
        GlobalInfo.__WriteDict(GlobalInfo.idToIdf, GlobalInfo.nameIdToIdf)
        GlobalInfo.__WriteDict(GlobalInfo.newIdToId, GlobalInfo.nameNewIdToId)
        return True
        
    @staticmethod
    def __ReadDict(dic, filename, typeK, typeV):
        f = open(filename, "r")
        for line in f:
            line = line.decode("utf-8")
            vec = line.split("\t")
            k = vec[0]
            v = vec[1]
            if (typeK == "int"):
                k = int(k)

            if (typeV == "int"):
                v = int(v)
            elif (typeV == "float"):
                v= float(v)

            dic[k] = v
        f.close()

    @staticmethod
    def __WriteDict(dic, filename):
        f = open(filename, "w")
        for k,v in dic.iteritems():
            if isinstance(k, (str, unicode)):
                f.write(k.encode("utf-8"))
            else:
                f.write(str(k))
            f.write("\t")
            if isinstance(v, (str, unicode)):
                f.write(v.encode("utf-8"))
            else:
                f.write(str(v))
            f.write("\n")
        f.close()
    
    @staticmethod
    def ReadDict(dic, nodeName):
        if (not GlobalInfo.isInit):
            print "init GlobalInfo before using"
        path = GlobalInfo.curNode.GetChild(nodeName).GetValue()
        GlobalInfo.__ReadDict(dic, path)

    @staticmethod
    def WriteDict(dic, nodeName):
        if (not GlobalInfo.isInit):
            print "init GlobalInfo before using"
        path = GlobalInfo.curNode.GetChild(nodeName).GetValue()
        GlobalInfo.__WriteDict(dic, path)
