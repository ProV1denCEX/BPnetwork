#include "BPNetwork.h"
#include <iomanip>

BPNetwork::BPNetwork(): n_learn_rate_in_(0), n_learn_rate_out_(0), n_input_nodes_(0), n_cycles_(0), n_size_io_(0),
                        n_hidden_layers_(0)
{
}

BPNetwork::BPNetwork(const std::string s_file_name, const double n_error_tolerance = 0,
	const double n_learn_rate_in = 0, const double n_learn_rate_out = 0, const int n_cycles = 0, const std::vector<int> d_hidden_nodes = { 0 }, const int n_size_io = 0)
{
	GetRawData(s_file_name);
	this->SetErrorTolerance(n_error_tolerance);
	this->SetLearnRateIn(n_learn_rate_in);
	this->SetLearnRateOut(n_learn_rate_out);
	this->SetCycles(n_cycles);
	this->SetHiddenNodes(d_hidden_nodes);
	this->SetSizeIO(n_size_io);
	this->SetInputNodes();
}

BPNetwork::~BPNetwork()
{
}

void BPNetwork::Train(const int n_start, const int n_end)
{
	InitializeLayers();
	GetDataPart(n_start, n_end, &d_train_data_);
	TrainModel();
	std::cout << "Training Complete !" << std::endl;
}

void BPNetwork::Test(const int n_start, const int n_end)
{
	GetDataPart(n_start, n_end, &d_test_data_);
	auto d_test_result = Predict();
	std::cout << "Prediction Results: " << std::endl;
	std::cout << "Actual  |  Result" << std::endl;
	for (auto i = 0; i < d_test_data_.size(); ++i)
	{
		for (auto j = 0; j < n_output_nodes_; ++j)
		{
			std::cout << std::setw(6) << d_test_data_[i][d_test_data_[0].size() - j - 1];
		}
		std::cout << "  |  ";
		for (auto j = 0; j < n_output_nodes_; ++j)
		{
			std::cout << std::setw(6) << d_test_result[i][j];
		}
		std::cout << std::endl;
	}
}

void BPNetwork::Save()
{
	// WEIGHT
	std::ofstream save_weight("BP_Network_Weight.txt", std::ios::out);
	
	// Input Nodes
	save_weight << "***Input Layer:\n";
	for (auto i = 0; i < n_input_nodes_; ++i)
	{
		save_weight << "*Node " << i << " : ";
		for (auto j = 0; j < input_layer[i].d_weight.size(); ++j)
		{
			if (j != input_layer[i].d_weight.size() - 1)
			{
				save_weight << input_layer[i].d_weight[j] << ", ";
			}
			else
			{
				save_weight << input_layer[i].d_weight[j];
			}
		}
		save_weight << "\n";
	}

	// Hidden Layers
	save_weight << "***Hidden Layer:\n";
	for (auto i = 0; i < n_hidden_layers_; ++i)
	{
		save_weight << "**Layer " << i << " : \n";
		for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
		{
			save_weight << "*Node " << j << " : ";
			for (auto k = 0; k < hidden_layer[i][j].d_weight.size(); ++k)
			{
				if (k != hidden_layer[i][j].d_weight.size() - 1)
				{
					save_weight << hidden_layer[i][j].d_weight[k] << ", ";
				}
				else
				{
					save_weight << hidden_layer[i][j].d_weight[k];
				}
				
			}
			save_weight << "\n";
		}
	}
	save_weight.close();

	// BIAS
	std::ofstream save_bias("BP_Network_Bias.txt", std::ios::out);

	// Hidden Layers
	save_bias << "***Hidden Layer:\n";
	for (auto i = 0; i < n_hidden_layers_; ++i)
	{
		save_bias << "**Layer " << i << " : \n";
		for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
		{
			save_bias << "*Node " << j << " : ";
			if (j != d_hidden_nodes_[i] - 1)
			{
				save_bias << hidden_layer[i][j].n_bias << ", ";
			}
			else
			{
				save_bias << hidden_layer[i][j].n_bias;
			}
			save_bias << "\n";
		}
	}

	save_bias << "***Output Layer:\n";
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		save_bias << "*Node " << i << " : ";
		if (i != output_layer.size() - 1)
		{
			save_bias << output_layer[i].n_bias << ", ";
		}
		else
		{
			save_bias << output_layer[i].n_bias;
		}
		save_bias << "\n";
	}
	save_bias.close();
}

void BPNetwork::TrainModel()
{
	auto n_learn_cycle = 0;
	while (n_error_now > this->n_error_tolerance_ &&
		n_learn_cycle < this->n_cycles_)
	{
		std::cout << "Training Error: " << n_error_now << std::endl;
		n_error_now = 0;
		InitializeLayersDelta();
		for (auto i = 0; i < d_train_data_.size(); ++i)
		{
			// Setting New Data
			for (auto j = 0; j < n_input_nodes_; ++j)
			{
				input_layer[j].n_value = d_train_data_[i][j + 1];
			}
			for (auto j = 0; j < n_output_nodes_; ++j)
			{
				output_layer[j].n_rightout = d_train_data_[i][d_train_data_[0].size() - j - 1]; 
			}

			// Epochs
			FPEpoch();
			BPEpoch();
		}

		// bp on input layer-> weight
		for (auto i = 0; i < input_layer.size(); ++i)
		{
			for (auto j = 0; j < d_hidden_nodes_[0]; ++j)
			{
				input_layer[i].d_weight[j] -= n_learn_rate_in_ * input_layer[i].d_delta_sum[j] / d_train_data_.size();
			}
		}

		// bp on hidden layer -> weight & bias -> update only, cal has done before
		for (auto i = 0; i < n_hidden_layers_; ++i)
		{
			if (i != n_hidden_layers_ - 1)
			{
				for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
				{
					hidden_layer[i][j].n_bias -= n_learn_rate_in_ * hidden_layer[i][j].n_delta_sum / d_train_data_.size();
					for (auto k = 0; k < d_hidden_nodes_[i + 1]; ++k)
					{
						hidden_layer[i][j].d_weight[k] -= n_learn_rate_in_ * hidden_layer[i][j].d_delta_sum[k] / d_train_data_.size();
					}
				}
			}
			else
			{
				for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
				{
					hidden_layer[i][j].n_bias -= n_learn_rate_out_ * hidden_layer[i][j].n_delta_sum / d_train_data_.size();
					for (auto k = 0; k < n_output_nodes_; ++k)
					{
						hidden_layer[i][j].d_weight[k] -= n_learn_rate_out_ * hidden_layer[i][j].d_delta_sum[k] / d_train_data_.size();
					}
				}
			}
		}

		// bp on output layer -> bias -> update only
		for (auto i = 0; i < n_output_nodes_; ++i)
		{
			output_layer[i].n_bias -= n_learn_rate_out_ * output_layer[i].n_delta_sum / d_train_data_.size();
		}

		// Update for stop loop
		++n_learn_cycle;
	}
}

std::vector<std::vector<double>> BPNetwork::Predict()
{
	// initialized output vector
	std::vector<std::vector<double>> d_predict_result;
	d_predict_result.resize(d_test_data_.size());
	for (auto i = 0; i < d_test_data_.size(); ++i)
	{
		for (auto j = 0; j < n_output_nodes_; ++j)
		{
			d_predict_result[i].push_back(0);
		}
	}

	// Start Predicition
	for (auto iData = 0; iData < d_test_data_.size(); ++iData)
	{
		// input new data
		for (auto i = 0; i < n_input_nodes_; ++i)
		{
			input_layer[i].n_value = d_test_data_[iData][i + 1];
		}

		// fp on hidden layer
		for (auto i = 0; i < n_hidden_layers_; ++i)
		{
			if (i != 0)
			{
				for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
				{
					double n_sum = 0;
					for (auto k = 0; k < d_hidden_nodes_[i - 1]; ++k)
					{
						n_sum += hidden_layer[i - 1][k].n_value * hidden_layer[i - 1][k].d_weight[j];
					}
					n_sum += hidden_layer[i][j].n_bias;
					hidden_layer[i][j].n_value = BioSigmoid(n_sum);
				}
			}
			else
			{
				for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
				{
					double n_sum = 0;
					for (auto k = 0; k < input_layer.size(); ++k)
					{
						n_sum += input_layer[k].n_value * input_layer[k].d_weight[j];
					}
					n_sum += hidden_layer[i][j].n_bias;
					hidden_layer[i][j].n_value = BioSigmoid(n_sum);
				}
			}
		}

		// fp on output layer
		for (auto i = 0; i < n_output_nodes_; ++i)
		{
			double n_sum = 0;
			for (auto j = 0; j < d_hidden_nodes_.back(); ++j)
			{
				n_sum += hidden_layer.back()[j].n_value * hidden_layer.back()[j].d_weight[i];
			}
			n_sum += output_layer[i].n_bias;
			output_layer[i].n_value = BioSigmoid(n_sum);
			d_predict_result[iData][i] = output_layer[i].n_value;
		}
	}
	return d_predict_result;
}

double BPNetwork::GetErrorTolerance() const
{
	return this->n_error_tolerance_;
}

double BPNetwork::GetLearnRateIn() const
{
	return this->n_learn_rate_in_;
}

double BPNetwork::GetLearnRateOut() const
{
	return this->n_learn_rate_out_;
}

std::vector<int> BPNetwork::GetHiddenNodes() const
{
	return this->d_hidden_nodes_;
}

std::vector<int> BPNetwork::GetHiddenLayersParam() const
{
	return d_hidden_nodes_;
}

std::vector<std::vector<double>> BPNetwork::GetActualLabel()
{
	std::vector<std::vector<double>> d_actual_label;
	d_actual_label.resize(d_test_data_.size());
	for (auto i = 0; i < d_test_data_.size(); ++i)
	{
		for (auto j = 0; j < n_output_nodes_; ++j)
		{
			d_actual_label[i].push_back(d_test_data_[i].at(d_test_data_[i].size() - j - 1));
		}
	}
	return d_actual_label;
}

int BPNetwork::GetInputNodes(int) const
{
	return this->n_input_nodes_;
}

int BPNetwork::GetCycles() const
{
	return this->n_cycles_;
}

int BPNetwork::GetSizeIO() const
{
	return this->n_size_io_;
}

int BPNetwork::GetSizeRawData() const
{
	return this->d_raw_data_.size() - 1;
}

void BPNetwork::SetErrorTolerance(const double n_new_num)
{
	if (n_new_num > 0)
	{
		this->n_error_tolerance_ = n_new_num;
	}
	else
	{
		std::cerr << "Illegal input num !" << std::endl;
	}
}

void BPNetwork::SetLearnRateIn(const double n_new_num)
{
	if (n_new_num > 0 && n_new_num < 1)
	{
		this->n_learn_rate_in_ = n_new_num;
	}
	else
	{
		std::cerr << "Illegal input num !" << std::endl;
	}
}

void BPNetwork::SetLearnRateOut(const double n_new_num)
{
	if (n_new_num > 0 && n_new_num < 1)
	{
		this->n_learn_rate_out_ = n_new_num;
	}
	else
	{
		std::cerr << "Illegal input num !" << std::endl;
	}
}

void BPNetwork::SetHiddenNodes(const std::vector<int> d_new_num)
{
	for (auto element : d_new_num)
	{
		if (element <= 0)
		{
			std::cerr << "Illegal input num !" << std::endl;
			return;
		}
	}
	d_hidden_nodes_ = d_new_num;
	this->n_hidden_layers_ = d_hidden_nodes_.size();
}

void BPNetwork::SetInputNodes()
{
	this->n_input_nodes_ = d_raw_data_[0].size() - 2;
}

void BPNetwork::SetCycles(const int n_new_num)
{
	if (n_new_num > 0)
	{
		this->n_cycles_ = n_new_num;
	}
	else
	{
		std::cerr << "Illegal input num !" << std::endl;
	}
}

void BPNetwork::SetSizeIO(const int n_new_num)
{
	if (n_new_num > 0)
	{
		this->n_size_io_ = n_new_num;
	}
	else
	{
		std::cerr << "Illegal input num !" << std::endl;
	}
}

/*
 * Do not know the col and row by default
 * Convert Date -> time_t
 * Convert Str - > double
 */
void BPNetwork::GetRawData(const std::string s_file_name)
{
	std::ifstream get_raw_data(s_file_name, std::ios::in);
	if (get_raw_data)
	{
		std::string s_data;
		std::stringstream ss;
		while (get_raw_data >> s_data)
		{
			if (s_data[0] < 0) continue;
			std::vector<std::string> c_result;
			this->split(s_data, ',', &c_result);
			if (this->c_factor_name_.empty())
			{
				this->c_factor_name_ = c_result;
			}
			else
			{
				std::vector<double> d_result(c_result.size());
				for (auto icol = 0; icol < c_result.size(); ++icol)
				{
					if (icol == 0)
					{
						d_result[icol] = this->DateNum(c_result[icol]);
					}
					else
					{
						d_result[icol] = std::stod(c_result[icol]);
					}
				}
				this->d_raw_data_.push_back(d_result);
			}
		}
	}
	else
	{
		std::cerr << "Cannot Find file: " << s_file_name << std::endl;
	}
	get_raw_data.close();
	NormalizeData(&d_raw_data_);
}

void BPNetwork::NormalizeData(std::vector<std::vector<double>>* d_raw_data)
{
	// copy a temp to cal
	std::vector<std::vector<double>> d_temp_B;
	d_temp_B.assign(d_raw_data->begin(), d_raw_data->end());

	// A to B
	double n_b_min = 0;
	double n_b_max = 0;
	for (auto icol = 1; icol < d_raw_data->at(0).size(); ++icol)
	{
		n_b_max = 0;
		n_b_min = 0;
		for (int i = d_raw_data->size() - 1; i >= 0; --i)
		{
			if (i != 0)
			{
				d_temp_B[i][icol] = 200 * (d_raw_data->at(i)[icol] - (d_raw_data->at(i - 1)[icol]))
					/ (d_raw_data->at(i)[icol] + d_raw_data->at(i - 1)[icol]);
			}
			else
			{
				d_temp_B[i][icol] = 0;
			}
			
			if (d_temp_B[i][icol] > n_b_max)
			{
				n_b_max = d_temp_B[i][icol];
			}

			if (d_temp_B[i][icol] < n_b_min)
			{
				n_b_min = d_temp_B[i][icol];
			}
		}

		// B to C
		for (auto i = 0; i < d_raw_data->size(); ++i)
		{
			d_raw_data->at(i)[icol] = 2 * ((d_temp_B[i][icol] - n_b_min) / (n_b_max - n_b_min)) - 1;
		}
	}
}

void BPNetwork::GetDataPart(int n_start, int n_end, std::vector<std::vector<double>>* d_data2write)
{
	if (n_start <= 0)
	{
		n_start = 0;
	}

	if (n_end >= d_raw_data_.size() || n_end <= n_start)
	{
		n_end = d_raw_data_.size();
	}
	d_data2write->clear();
	d_data2write->assign(d_raw_data_.begin() + n_start, d_raw_data_.begin() + n_end);
}

/*
 * Split Line by delim
 */
void BPNetwork::split(const std::string &str_2_split, const char delim, std::vector<std::string> *elems, const bool skip_empty) const
{
	std::istringstream iss(str_2_split);
	for (std::string item; getline(iss, item, delim); )
	{
		if (skip_empty && item.empty())
		{
			continue;
		}
		elems->push_back(item);
	}
}

/*
 * Convert Date 2 time_t
 */
time_t BPNetwork::DateNum(std::string time2convert)
{

	// Convert tm to time_t
	struct tm temp;
	time_t tt;
	localtime_s(&temp, &tt);

	sscanf_s(time2convert.c_str(), "%d/%d/%d",
		&temp.tm_year, &temp.tm_mon, &temp.tm_mday);
	temp.tm_mon--;
	temp.tm_year -= 1900;
	// change time
	return mktime(&temp);
}

/*
 * Initialize all layers
 */
void BPNetwork::InitializeLayers()
{
	// input layer
	input_layer.clear();
	input_layer.resize(this->n_input_nodes_);
	for (auto i = 0; i < n_input_nodes_; ++i)
	{
		for (auto j = 0; j < d_hidden_nodes_[0]; ++j)
		{
			input_layer[i].d_weight.push_back(GetRand());
			input_layer[i].d_delta_sum.push_back(0);
		}
	}

	// Output layer
	output_layer.clear();
	output_layer.resize(this->n_output_nodes_);
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		output_layer[i].n_bias = GetRand();
	}

	// hidden layers
	hidden_layer.clear();
	hidden_layer.resize(this->n_hidden_layers_);
	for (auto i = 0; i < n_hidden_layers_; ++i)
	{
		hidden_layer[i].resize(d_hidden_nodes_[i]);
		for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
		{
			hidden_layer[i][j].n_bias = GetRand();
			if (i != n_hidden_layers_ - 1)
			{
				for (auto k = 0; k < d_hidden_nodes_[i + 1]; ++k)
				{
					hidden_layer[i][j].d_weight.push_back(GetRand());
					hidden_layer[i][j].d_delta_sum.push_back(0);
				}
			}
			else
			{
				for (auto k = 0; k < output_layer.size(); ++k)
				{
					hidden_layer[i][j].d_weight.push_back(GetRand());
					hidden_layer[i][j].d_delta_sum.push_back(0);
				}
			}
		}
	}
}

void BPNetwork::InitializeLayersDelta()
{
	// input layer
	for (auto i = 0; i < n_input_nodes_; ++i)
	{
		input_layer[i].d_delta_sum.assign(input_layer[i].d_delta_sum.size(), 0);
	}

	// hidden layers
	for (auto i = 0; i < hidden_layer.size(); ++i)
	{
		for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
		{
			hidden_layer[i][j].n_delta_sum = 0;
			hidden_layer[i][j].d_delta_sum.assign(hidden_layer[i][j].d_delta_sum.size(), 0);
		}
	}

	// Output layer
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		output_layer[i].n_delta_sum = 0;
		output_layer[i].n_delta = 0;
	}
}

/*
 * FP process
 */
void BPNetwork::FPEpoch()
{
	// forward propagation on hidden layers
	for (auto i = 0; i < this->n_hidden_layers_; ++i)
	{
		if (i != 0)
		{
			for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
			{
				double n_sum = 0;
				for (auto k = 0; k < d_hidden_nodes_[i - 1]; ++k)
				{
					n_sum += hidden_layer[i - 1][k].n_value * hidden_layer[i - 1][k].d_weight[j];
				}
				n_sum += hidden_layer[i][j].n_bias;
				hidden_layer[i][j].n_value = BioSigmoid(n_sum);
			}
		}
		else
		{
			for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
			{
				double n_sum = 0;
				for (auto k = 0; k < input_layer.size(); ++k)
				{
					n_sum += input_layer[k].n_value * input_layer[k].d_weight[j];
				}
				n_sum += hidden_layer[i][j].n_bias;
				hidden_layer[i][j].n_value = BioSigmoid(n_sum);
			}
		}
	}

	// fp on output layer
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		double n_sum = 0;
		for (auto j = 0; j < d_hidden_nodes_.back(); ++j)
		{
			n_sum += hidden_layer.back()[j].n_value * hidden_layer.back()[j].d_weight[i];
		}
		n_sum += output_layer[i].n_bias;
		output_layer[i].n_value = BioSigmoid(n_sum);
	}
}

void BPNetwork::BPEpoch()
{
	// bp on output layer
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		const auto n_temp = fabs(output_layer[i].n_value - output_layer[i].n_rightout);
		n_error_now += n_temp * n_temp / 2;
		output_layer[i].n_delta = (output_layer[i].n_value - output_layer[i].n_rightout) *
			(1 - output_layer[i].n_value) * (1 + output_layer[i].n_value) * 0.5;
	}

	// bp on hidden layer -> delta
	for (auto i = n_hidden_layers_ - 1; i >= 0; --i)
	{
		if (i != n_hidden_layers_ - 1)
		{
			for (auto j = 0; j < hidden_layer[i].size(); ++j)
			{
				double n_sum = 0;
				for (auto k = 0; k < hidden_layer[i + 1].size(); ++k)
				{
					n_sum += hidden_layer[i + 1][k].n_delta * hidden_layer[i][j].d_weight[k];
					hidden_layer[i][j].n_delta = n_sum * (1 - hidden_layer[i][j].n_value) * (1 + hidden_layer[i][j].n_value) * 0.5;
				}
			}
		}
		else
		{
			for (auto j = 0; j < hidden_layer[i].size(); ++j)
			{
				double n_sum = 0;
				for (auto k = 0; k < output_layer.size(); ++k)
				{
					n_sum += output_layer[k].n_delta * hidden_layer[i][j].d_weight[k];
					hidden_layer[i][j].n_delta = n_sum * (1 - hidden_layer[i][j].n_value) * (1 + hidden_layer[i][j].n_value) * 0.5;
				}
			}
		}
	}

	// bp on input layer -> delta, weight
	for (auto i = 0; i < n_input_nodes_; ++i)
	{
		for (auto j = 0; j < hidden_layer[0].size(); ++j)
		{
			input_layer[i].d_delta_sum[j] += input_layer[i].n_value * hidden_layer[0][j].n_delta;
		}
	}

	// bp on hidden layer -> weight change sum & bias change sum
	for (auto i = 0; i < n_hidden_layers_; ++i)
	{
		if (i != n_hidden_layers_ - 1)
		{
			for (auto j = 0; j < d_hidden_nodes_[i]; ++j)
			{
				hidden_layer[i][j].n_delta_sum += hidden_layer[i][j].n_delta;
				for (auto k = 0; k < d_hidden_nodes_[i + 1]; ++k)
				{
					hidden_layer[i][j].d_delta_sum[k] += hidden_layer[i][j].n_value * hidden_layer[i + 1][k].n_delta;
				}
			}
		}
		else
		{
			for (auto j = 0; j < d_hidden_nodes_.back(); ++j)
			{
				hidden_layer[i][j].n_delta_sum += hidden_layer[i][j].n_delta;
				for (auto k = 0; k < output_layer.size(); ++k)
				{
					hidden_layer[i][j].d_delta_sum[k] += hidden_layer[i][j].n_value * output_layer[k].n_delta;
				}
			}
		}
	}

	// bp on output layer -> bias delta sum
	for (auto i = 0; i < n_output_nodes_; ++i)
	{
		output_layer[i].n_delta_sum += output_layer[i].n_delta;
	}
}
