#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <time.h>
#include <chrono>
#include <algorithm>

inline double GetRand()
{
	return ((2.0 * static_cast<double>(rand()) / RAND_MAX) - 1);
}

inline double BioSigmoid(const double a)
{
	return 2 / (1 + exp(-a)) - 1;
}


/**
 * \brief Input Nodes
 * There are:
 * 1. value -> input value
 * 2. d_weight -> weight vector to the first hidden layer
 * 3. d_delta_sum -> accumulated delta to the firste hidden layer
 */
struct InputNode
{
	double n_value;
	std::vector<double> d_weight, d_delta_sum;
};

/*
 * Output Nodes
 * 1. n_value -> output value (label)
 * 2. delta -> output value - currect value
 * 3. rightout -> currect value
 * 4. bias -> shift
 * 5. n_delta_sum -> accumulated delta of bias
 */
struct OutputNode
{
	double n_value, n_delta, n_rightout, n_bias, n_delta_sum;
};


/*
 * 1. value -> value of current node
 * 2. delta -> delta obtained by bp
 * 3. bias -> shift
 * 4. n_delta_sum -> accumulated delta of bias
 * 5. weight -> weight to next layer
 * 6. d_delta_sum -> accumulated delta of weight
 */
struct HiddenNode
{
	double n_value, n_delta, n_bias, n_delta_sum;
	std::vector<double> d_weight, d_delta_sum;
};


class BPNetwork
{
public:
	BPNetwork();
	BPNetwork(std::string, double, double, double, int, std::vector<int>, int);
	~BPNetwork();
	void Train(int, int);
	void Test(int, int);
	void Save();

	double GetErrorTolerance() const;
	double GetLearnRateIn() const;
	double GetLearnRateOut() const;
	std::vector<int> GetHiddenNodes() const;
	std::vector<int> GetHiddenLayersParam() const;
	std::vector<std::vector<double>> GetActualLabel();
	int GetInputNodes(int) const;
	int GetCycles() const;
	int GetSizeIO() const;
	int GetSizeRawData() const;

	void SetErrorTolerance(double);
	void SetLearnRateIn(double);
	void SetLearnRateOut(double);
	void SetHiddenNodes(std::vector<int>);
	void SetInputNodes();
	void SetCycles(int);
	void SetSizeIO(int);

private:
	void GetRawData(std::string);
	static void NormalizeData(std::vector<std::vector<double>>*);
	void GetDataPart(int, int, std::vector<std::vector<double>>*);
	void split(const std::string &, const char , std::vector<std::string> *, const bool = true) const;
	void InitializeLayers();
	void InitializeLayersDelta();
	void TrainModel();
	void FPEpoch();
	void BPEpoch();
	std::vector<std::vector<double>> Predict();
	std::vector<std::vector<double>> d_raw_data_;
	std::vector<std::vector<double>> d_train_data_;
	std::vector<std::vector<double>> d_test_data_;

	std::vector<std::string> c_factor_name_;
	std::vector<int> d_hidden_nodes_;

	static time_t DateNum(std::string);
	double n_error_tolerance_ = 0;
	double n_learn_rate_in_;
	double n_learn_rate_out_;
	double n_error_now = RAND_MAX;
	int n_input_nodes_;
	int n_output_nodes_ = 1;
	int n_cycles_;
	int n_size_io_;
	int n_hidden_layers_;

	std::vector<InputNode> input_layer;
	std::vector<OutputNode> output_layer;
	std::vector<std::vector<HiddenNode>> hidden_layer;
};
