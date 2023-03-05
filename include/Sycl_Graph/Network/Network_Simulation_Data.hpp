#ifndef SYCL_GRAPH_NETWORK_SIMULATION_DATA_HPP
#define SYCL_GRAPH_NETWORK_SIMULATION_DATA_HPP

namespace Sycl_Graph::Network
{

    template <typename MeasureState, typename NodeState, typename EdgeState>
    struct Trajectory
    {
        Trajectory(uint32_t Nt = 0): measure(Nt), nodes(Nt), edges(Nt){}
        std::vector<MeasureState> measure;
        std::vector<std::vector<NodeState>> nodes;
        std::vector<std::vector<EdgeState>> edges;
    };

    template <typename MeasureState, typename NodeState>
    struct Trajectory<MeasureState, NodeState, void>
    {
        Trajectory(uint32_t Nt = 0): measure(Nt), nodes(Nt){}
        std::vector<MeasureState> measure;
        std::vector<std::vector<NodeState>> nodes;
    };

    template <typename MeasureState>
    struct Trajectory<MeasureState, void, void>
    {
        Trajectory(uint32_t Nt = 0): measure(Nt){}
        std::vector<MeasureState> measure;
    };

    template <typename MeasureState, typename NodeState, typename EdgeState>
    struct Simulation_Data
    {
        Simulation_Data(uint32_t Nt = 0): data(Nt){}
        Trajectory<MeasureState, NodeState, EdgeState> data;

        void write(std::string filename, const std::vector<std::string>& colnames, std::string delimiter = ",")
        {
            std::ofstream file;
            file.open(filename);
            for (int i = 0; i < colnames.size(); i++)
            {
                file << colnames[i];
                if (i < colnames.size() - 1)
                    file << delimiter;
            }
            file << std::endl;
            for (int i = 0; i < trajectory.size(); i++)
            {
                file << data.measure[i]<< delimiter;
                std::for_each(data.nodes[i].begin(), data.nodes[i].end(), [&](auto& x){file << x << delimiter;});
                std::for_each(data.edges[i].begin(), data.edges[i].end(), [&](auto& x){file << x << delimiter;});
                if (i < trajectory.size() - 1)
                    file << delimiter;
            }
            file.close();
        }

        template <typename uI_t>
        void write(std::string filename, const std::vector<uI_t>& ids, std::string delimiter = ",")
        {
            std::vector<std::string> colnames = {"State"};
            for (int i = 0; i < ids.size(); i++)
            {
                colnames.push_back(std::to_string(ids[i]));
            }
            write(filename, colnames, delimiter);
        }
    };

    template <typename MeasureState, typename NodeState>
}


#endif