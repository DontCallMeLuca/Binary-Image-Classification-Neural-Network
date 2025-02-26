#pragma once

#include <torch/torch.h>
#include <filesystem>
#include <vector>
#include <string>
#include <random>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class Dataset : public torch::data::Dataset<Dataset>
{
private:
	constexpr static const int	image_size_ = 256;
	std::vector<std::string>	image_paths_;
	std::vector<int>			labels_;

	bool is_valid_image(const std::string& path)
	{
		std::string ext = fs::path(path).extension().string();
		return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
	}

	torch::Tensor load_image(const std::string& path)
	{
		cv::Mat img = cv::imread(path);
		if (img.empty())
			throw std::runtime_error("Error loading image: " + path);
		cv::resize(img, img, cv::Size(image_size_, image_size_));
		torch::Tensor tensor_image = torch::from_blob(
			img.data, { img.rows, img.cols, 3 }, torch::kByte
		);

		return tensor_image.permute({ 2, 0, 1 }).to(torch::kFloat32).div(255.0);
	}

public:
	Dataset(void) = delete;
	~Dataset(void) = default;

	Dataset(const std::string& categoryA_path, const std::string& categoryB_path)
	{
		for (const auto& entry : fs::directory_iterator(categoryA_path))
		{
			if (is_valid_image(entry.path().string()))
			{
				image_paths_.push_back(entry.path().string());
				labels_.push_back(0);
			}
		}
		for (const auto& entry : fs::directory_iterator(categoryB_path))
		{
			if (is_valid_image(entry.path().string()))
			{
				image_paths_.push_back(entry.path().string());
				labels_.push_back(1);
			}
		}

		std::vector<size_t> indices(image_paths_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);

		std::vector<std::string> shuffled_paths;
		std::vector<int> shuffled_labels;

		for (size_t idx : indices)
		{
			shuffled_paths.push_back(image_paths_[idx]);
			shuffled_labels.push_back(labels_[idx]);
		}

		image_paths_ = shuffled_paths;
		labels_ = shuffled_labels;
	}

	inline torch::data::Example<> get(size_t index) override
	{
		torch::Tensor image = load_image(image_paths_[index]);
		torch::Tensor label = torch::tensor(labels_[index], torch::kFloat32);
		return { image, label };
	}

	inline torch::optional<size_t> size() const noexcept override
	{
		return image_paths_.size();
	}
};
