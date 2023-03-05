
const char* filepath = "../data/sentrum_grid/data.gml";


#include "pugixml.hpp"
#include <iostream>
int main()
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filepath);

    //list children
    for (pugi::xml_node child : doc.child("graphml").children())
    {
        std::cout << child.name() << std::endl;
    }    

    if (!result)
        std::cout << "Error: " << result.description() << ", character pos = " << result.offset << "\n";
        return -1;
        
}