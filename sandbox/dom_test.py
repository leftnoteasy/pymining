from xml.dom import minidom

pxml = minidom.parse("test.xml")
print pxml.getElementsByTagName("cfg")
pnode = pxml.firstChild
for node in pnode.getElementsByTagName("a"):
    print node.firstChild.data

