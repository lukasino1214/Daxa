#pragma once

#include "../DaxaCore.hpp"

#include <optional>

#include <vulkan/vulkan.h>

#include <VkBootstrap.h>


namespace daxa {
	namespace gpu {

		class Instance {
		public:
			Instance(char const* AppName = "Daxa Application", char const* EngineName = "Daxa", bool enableValidationLayer = true);
			~Instance();

			VkInstance getVkInstance() const { return instance.instance; }
			vkb::Instance& getVKBInstance() { return instance; }
		private:
			vkb::Instance instance;
		};

		extern std::unique_ptr<Instance> instance;

	}
}