#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <vector>
#include <string>
#include <random>

#include "Dataset.hpp"

namespace Model
{
	namespace fs = std::filesystem;

	struct CNNImpl : torch::nn::Module
	{
		CNNImpl()
		{
			conv1 = register_module("conv1",
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1)));
			pool1 = register_module("pool1",
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

			conv2 = register_module("conv2",
				torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1)));
			pool2 = register_module("pool2",
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

			conv3 = register_module("conv3",
				torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).stride(1)));
			pool3 = register_module("pool3",
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

			constexpr const int dense_input_features = 16 * 30 * 30;
			fc1 = register_module("fc1", torch::nn::Linear(dense_input_features, 256));
			fc2 = register_module("fc2", torch::nn::Linear(256, 1));
		}

		torch::Tensor forward(torch::Tensor x)
		{
			x = torch::relu(conv1->forward(x));
			x = pool1->forward(x);
			x = torch::relu(conv2->forward(x));
			x = pool2->forward(x);
			x = torch::relu(conv3->forward(x));
			x = pool3->forward(x);

			x = x.view({ x.size(0), -1 });
			x = torch::relu(fc1->forward(x));
			x = torch::sigmoid(fc2->forward(x));
			return x;
		}

		torch::nn::Conv2d       conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
		torch::nn::MaxPool2d    pool1{ nullptr }, pool2{ nullptr }, pool3{ nullptr };
		torch::nn::Linear       fc1{ nullptr }, fc2{ nullptr };
	};

	TORCH_MODULE(CNN);

	void train(CNN& model, Dataset& dataset, int batch_size, int epochs)
	{
		auto total_size = dataset.size().value();
		auto train_size = static_cast<size_t>(total_size * 0.7);
		auto val_size = static_cast<size_t>(total_size * 0.2);

		std::vector<size_t> indices(total_size);
		std::iota(indices.begin(), indices.end(), 0);
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);

		std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
		std::vector<size_t> val_indices(indices.begin() + train_size, indices.begin() + train_size + val_size);

		auto train_sampler = torch::data::samplers::RandomSampler(train_size);
		auto val_sampler = torch::data::samplers::SequentialSampler(val_size);

		auto train_loader = torch::data::make_data_loader<
			torch::data::samplers::RandomSampler>(
				dataset, train_sampler, batch_size);

		auto val_loader = torch::data::make_data_loader<
			torch::data::samplers::SequentialSampler>(
			dataset, val_sampler, batch_size);

		torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			model->train();
			float train_loss = 0.0;
			int batch_count = 0;

			for (auto& batch : *train_loader)
			{
				optimizer.zero_grad();

				std::vector<torch::Tensor> images_vec;
				std::vector<torch::Tensor> labels_vec;

				for (const auto& example : batch)
				{
					images_vec.push_back(example.data);
					labels_vec.push_back(example.target.reshape({ 1 }));
				}

				torch::Tensor images = torch::stack(images_vec);
				torch::Tensor labels = torch::stack(labels_vec).to(torch::kFloat32);

				torch::Tensor output = model->forward(images);
				torch::Tensor loss = torch::binary_cross_entropy(output, labels);

				loss.backward();
				optimizer.step();

				train_loss += loss.item<float>();
				batch_count++;
			}

			std::cout << "Epoch: " << epoch + 1 << ", Loss: " << (train_loss / batch_count) << std::endl;

			if (epoch % 5 == 0)
			{
				model->eval();
				float val_loss = 0.0;
				int val_batch_count = 0;

				torch::NoGradGuard no_grad;

				for (auto& batch : *val_loader)
				{
					std::vector<torch::Tensor> images_vec;
					std::vector<torch::Tensor> labels_vec;

					for (const auto& example : batch)
					{
						images_vec.push_back(example.data);
						labels_vec.push_back(example.target.reshape({ 1 }));
					}

					torch::Tensor images = torch::stack(images_vec);
					torch::Tensor labels = torch::stack(labels_vec).to(torch::kFloat32);

					torch::Tensor output = model->forward(images);
					torch::Tensor loss = torch::binary_cross_entropy(output, labels);

					val_loss += loss.item<float>();
					++val_batch_count;
				}

				std::cout << "Validation Loss: " << (val_loss / val_batch_count) << std::endl;
				model->train();
			}
		}
		torch::save(model, "trained/model.pt");
	}
}
