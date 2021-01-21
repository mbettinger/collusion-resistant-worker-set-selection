def formatVertexIds(file_path, new_file_path, header=False):
    """ We expect a file containing row of the format
        vertexId delemiter vertexId
    """
    delimiters={"csv":",","txt":" "}
    fileFormat=file_path.split(".")[-1]
    if fileFormat not in delimiters.keys():
        delimiter=" "
    else:
        delimiter=delimiters[fileFormat]
        
    with open(new_file_path,'w') as formatted_file:
        with open(file_path) as old_file:
            for line in old_file:
                if header:
                    header=False
                else:
                    columns = line.split(delimiter)
                    formatted_file.write("V" + columns[0] + " " + "V" + columns[1])
                    
from os import listdir
from os.path import isfile, join
def getFileNamesInDir(dirPath):
    onlyFiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    return onlyFiles