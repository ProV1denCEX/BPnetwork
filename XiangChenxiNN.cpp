#include "BPNetwork.h"

#include <filesystem>
#include <Windows.h>
#include <locale>

int main(int argc, char* argv[])
{
	std::cout << "Welcome to our system" << std::endl;
	std::cout << "	Team: Peiwen Zhang, Siyan Jin, Chenxi Xiang" << std::endl;
	std::cout << "********************************************" << std::endl;
	std::cout << "Attention: Our system only support csv file as source data file" << std::endl;
	std::cout << "Start Scanning csv files in current directory" << std::endl;

	// get current path
	char buffer[MAX_PATH];
	GetModuleFileName(nullptr, buffer, MAX_PATH);
	const auto pos = std::string(buffer).find_last_of("\\/");
	const auto path = std::string(buffer).substr(0, pos);

	// get all csv files
	std::vector<std::string> files;
	auto counter = 0;
	for (auto& ifile : std::experimental::filesystem::directory_iterator(path))
	{
		auto s_extension = ifile.path().extension();
		if (s_extension == ".csv" || s_extension == ".CSV")
		{
			files.push_back(ifile.path().string());
			std::cout << counter << ". " << ifile.path().string() << std::endl;
			++counter;
		}
	}

	auto is_net = false;
	auto n_mode = 0;
	BPNetwork bp_network;
	std::string s_file_name;
	while (true)
	{
		std::cout << "Please select : " << std::endl;
		std::cout << "	1. Network Generation " << std::endl;
		std::cout << "	2. Network Training " << std::endl;
		std::cout << "	3. Network Testing " << std::endl;
		std::cout << "	4. Save the Net" << std::endl;
		std::cout << "	5. Change Params of your Net" << std::endl;
		std::cout << "	6. Quit " << std::endl;
		std::cin >> n_mode;
		switch (n_mode)
		{
		case 1:
		{
			// Select file
			std::cout << "Enter number to select data source file: " << "0 to " << counter - 1 << std::endl;
			int n_file;
			std::cin >> n_file;

			while (n_file < 0 || n_file > files.size() - 1)
			{
				std::cout << "Please Enter number to select data source file : " << "0 to " << counter - 1 << std::endl;
				std::cin >> n_file;
			}
			s_file_name = files[n_file];

			// enter parms
			std::cout << "Please enter the error tolerance: " << std::endl;
			double n_error_tolerance;
			std::cin >> n_error_tolerance;

			std::cout << "Please enter the learn rate of input: " << std::endl;
			double n_learn_rate_in;
			std::cin >> n_learn_rate_in;

			std::cout << "Please enter the learn rate of output: " << std::endl;
			double n_learn_rate_out;
			std::cin >> n_learn_rate_out;

			std::cout << "Please enter the max cycles during training: " << std::endl;
			int n_cycles;
			std::cin >> n_cycles;

			std::cout << "Please enter the number of hidden layers: " << std::endl;
			int n_hidden_layers;
			std::cin >> n_hidden_layers;
			while (n_hidden_layers < 0)
			{
				std::cout << "Please enter the number of hidden layers : " << std::endl;
				std::cin >> n_hidden_layers;
			}
			std::vector<int> d_hidden_layers;
			int n_hidden_nodes;
			for (auto i = 0; i < n_hidden_layers; ++i)
			{
				std::cout << "Please enter hidden nodes number of Layer " << i << " : " << std::endl;
				std::cin >> n_hidden_nodes;
				while (n_hidden_nodes < 0)
				{
					std::cout << "Please enter hidden nodes number of Layer " << i << " : " << std::endl;
					std::cin >> n_hidden_nodes;
				}
				d_hidden_layers.push_back(n_hidden_nodes);
			}

			// Model Generation
			bp_network = BPNetwork(s_file_name, n_error_tolerance, n_learn_rate_in, n_learn_rate_out, n_cycles,
				d_hidden_layers, 1);
			is_net = true;
			break;
		}

		case 2:
		{
			if (is_net)
			{
				std::cout << "Please Enter the start point of train set in raw data set: " 
				<< "0 to " << bp_network.GetSizeRawData() << std::endl;
				int n_start;
				std::cin >> n_start;
				std::cout << "Please Enter the end point of train set in raw data set: " 
					<< n_start + 1 <<" to " << bp_network.GetSizeRawData() << std::endl;
				int n_end;
				std::cin >> n_end;
				bp_network.Train(n_start, n_end);
			}
			else
			{
				std::cout << "You should generate a net first !" << std::endl;
			}
			break;
		}

		case 3:
		{
			if (is_net)
			{
				std::cout << "Please Enter the start point of test set in raw data set: " << std::endl;
				int n_start;
				std::cin >> n_start;
				std::cout << "Please Enter the end point of test set in raw data set: " << std::endl;
				int n_end;
				std::cin >> n_end;
				bp_network.Test(n_start, n_end);
			}
			else
			{
				std::cout << "You should generate a net first !" << std::endl;
			}
			break;
		}

		case 4:
		{
			if (is_net)
			{
				bp_network.Save();
			}
			else
			{
				std::cout << "You should generate a net first !" << std::endl;
			}
			break;
		}

		case 5:
		{
			if (!is_net)
			{
				std::cout << "You should generate a net first !" << std::endl;
				break;
			}
			std::cout << "Please select param to change : " << std::endl;
			std::cout << "	2. error tolerance " << std::endl;
			std::cout << "	3. learning rate of input " << std::endl;
			std::cout << "	4. learning rate of output" << std::endl;
			std::cout << "	5. learning cycle limitation" << std::endl;
			std::cout << "	6. hidden layers and nodes " << std::endl;
			std::cout << "	7. Back " << std::endl;
			int n_param;
			std::cin >> n_param;
			if (n_param == 7)
			{
				break;
			}

			if (n_param == 6)
			{
				std::cout << "Please enter the number of hidden layers: " << std::endl;
				int n_hidden_layers;
				std::cin >> n_hidden_layers;
				while (n_hidden_layers < 0)
				{
					std::cout << "Please enter the number of hidden layers : " << std::endl;
					std::cin >> n_hidden_layers;
				}
				std::vector<int> d_hidden_layers;
				int n_hidden_nodes;
				for (auto i = 0; i < n_hidden_layers; ++i)
				{
					std::cout << "Please enter hidden nodes number of Layer " << i << " : " << std::endl;
					std::cin >> n_hidden_nodes;
					while (n_hidden_nodes < 0)
					{
						std::cout << "Please enter hidden nodes number of Layer " << i << " : " << std::endl;
						std::cin >> n_hidden_nodes;
					}
					d_hidden_layers.push_back(n_hidden_nodes);
				}
				bp_network.SetHiddenNodes(d_hidden_layers);
			}
			else
			{
				std::cout << "Please enter the new param" << std::endl;
				double n_new_param;
				std::cin >> n_new_param;

				switch (n_param)
				{
				case 2:
					bp_network.SetErrorTolerance(n_new_param);
					break;
				case 3:
					bp_network.SetLearnRateIn(n_new_param);
					break;
				case 4:
					bp_network.SetLearnRateOut(n_new_param);
					break;
				case 5:
					bp_network.SetCycles(n_new_param);
					break;
				default:;
				}
			}

			// re construct to init all private var -> Stupid, dangerous but easy method
			bp_network = BPNetwork(s_file_name, bp_network.GetErrorTolerance(), 
				bp_network.GetLearnRateIn(), bp_network.GetLearnRateOut(), bp_network.GetCycles(),
				bp_network.GetHiddenLayersParam(), 1);

			break;
		}

		case 6:
			return 0;

		default:
			// For testing
			is_net = true;
			s_file_name = files[0];
			const std::vector<int> d_layers(2, 10);
			bp_network = BPNetwork("Combined data.csv", 0.1, 0.95, 0.9, 100,
				d_layers, 1);
			bp_network.Train(0, 50);
			bp_network.Test(51, 54);
			break;
		}
	}
}
