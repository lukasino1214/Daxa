#include "impl_core.hpp"

#include "impl_gpu_resources.hpp"

#include <daxa/daxa.inl>
#include <format>

namespace daxa
{
    auto GPUResourceId::is_empty() const -> bool
    {
        return version == 0;
    }

    auto ImageId::default_view() const -> ImageViewId
    {
        return ImageViewId{{.index = index, .version = version}};
    }

    auto to_string(GPUResourceId const & id) -> std::string
    {
        return std::format("index: {}, version: {}", static_cast<u32>(id.index), static_cast<u32>(id.version));
    }

    auto to_string(ImageViewType const & type) -> std::string_view
    {
        switch (type)
        {
        case ImageViewType::REGULAR_1D: return "REGULAR_1D";
        case ImageViewType::REGULAR_2D: return "REGULAR_2D";
        case ImageViewType::REGULAR_3D: return "REGULAR_3D";
        case ImageViewType::CUBE: return "CUBE";
        case ImageViewType::REGULAR_1D_ARRAY: return "REGULAR_1D_ARRAY";
        case ImageViewType::REGULAR_2D_ARRAY: return "REGULAR_2D_ARRAY";
        case ImageViewType::CUBE_ARRAY: return "CUBE_ARRAY";
        default: return "NONE";
        }
    }

    auto GPUShaderResourceTable::initialize(u32 max_buffers,
            u32 max_images,
            u32 max_samplers,
            u32 max_acceleration_structures,
            VkPhysicalDevice physical_device,
            VkDevice device,
            VkBuffer device_address_buffer,
            VmaAllocator vma_allocator,
            Optional<DescriptorHeapProperties> descriptor_heap_properties_opt,
            PFN_vkGetPhysicalDeviceDescriptorSizeEXT vkGetPhysicalDeviceDescriptorSizeEXT,
            PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT) -> daxa_Result
    {
        daxa_Result result = DAXA_RESULT_SUCCESS;
        defer
        {
            if (result != DAXA_RESULT_SUCCESS)
            {
                if (this->vk_descriptor_pool)
                {
                    vkDestroyDescriptorPool(device, this->vk_descriptor_pool, nullptr);
                }
                if (this->vk_descriptor_set_layout)
                {
                    vkDestroyDescriptorSetLayout(device, this->vk_descriptor_set_layout, nullptr);
                }
            }
        };

        bool const ray_tracing_enabled = max_acceleration_structures != (~0u);

        auto round_up_to_pages = [](auto size, auto block_size){
            return (size + block_size - 1) / block_size * block_size;
        };

        u32 max_tlas = 1024; // TODO(Raytracing): Should we have a smarter limit?
        u32 max_blas = max_acceleration_structures;
        max_tlas = round_up_to_pages(max_tlas, static_cast<u32>(GpuResourcePool<>::PAGE_SIZE));
        max_blas = round_up_to_pages(max_blas, static_cast<u32>(GpuResourcePool<>::PAGE_SIZE));
        max_buffers = round_up_to_pages(max_buffers, static_cast<u32>(GpuResourcePool<>::PAGE_SIZE));
        max_images = round_up_to_pages(max_images, static_cast<u32>(GpuResourcePool<>::PAGE_SIZE));
        max_samplers = round_up_to_pages(max_samplers, static_cast<u32>(GpuResourcePool<>::PAGE_SIZE));

        buffer_slots.max_resources = max_buffers;
        image_slots.max_resources = max_images;
        sampler_slots.max_resources = max_samplers;
        if (ray_tracing_enabled)
        {
            tlas_slots.max_resources = max_tlas;
            blas_slots.max_resources = max_blas;
        }

        buffer_slots.hot_data = decltype(buffer_slots.hot_data)(buffer_slots.max_resources);
        image_slots.hot_data = decltype(image_slots.hot_data)(image_slots.max_resources);
        sampler_slots.hot_data = decltype(sampler_slots.hot_data)(sampler_slots.max_resources);
        if (ray_tracing_enabled)
        {
            tlas_slots.hot_data = decltype(tlas_slots.hot_data)(tlas_slots.max_resources);
            blas_slots.hot_data = decltype(blas_slots.hot_data)(blas_slots.max_resources);
        }

        if(!descriptor_heap_properties_opt.has_value())
        {
            VkDescriptorPoolSize const buffer_descriptor_pool_size{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = buffer_slots.max_resources + 1,
            };

            VkDescriptorPoolSize const storage_image_descriptor_pool_size{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = image_slots.max_resources,
            };

            VkDescriptorPoolSize const sampled_image_descriptor_pool_size{
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = image_slots.max_resources,
            };

            VkDescriptorPoolSize const sampler_descriptor_pool_size{
                .type = VK_DESCRIPTOR_TYPE_SAMPLER,
                .descriptorCount = sampler_slots.max_resources,
            };

            auto pool_sizes = std::vector{
                buffer_descriptor_pool_size,
                storage_image_descriptor_pool_size,
                sampled_image_descriptor_pool_size,
                sampler_descriptor_pool_size,
            };
            if (ray_tracing_enabled)
            {
                VkDescriptorPoolSize const as_descriptor_pool_size{
                    .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    .descriptorCount = tlas_slots.max_resources,
                };
                pool_sizes.push_back(as_descriptor_pool_size);
            }

            VkDescriptorPoolCreateInfo const vk_descriptor_pool_create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                .maxSets = 1,
                .poolSizeCount = static_cast<u32>(pool_sizes.size()),
                .pPoolSizes = pool_sizes.data(),
            };

            result = static_cast<daxa_Result>(vkCreateDescriptorPool(device, &vk_descriptor_pool_create_info, nullptr, &this->vk_descriptor_pool));
            _DAXA_RETURN_IF_ERROR(result, result)

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto const * descriptor_pool_name = "mega descriptor pool";
                VkDebugUtilsObjectNameInfoEXT const descriptor_pool_name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                    .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_pool),
                    .pObjectName = descriptor_pool_name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &descriptor_pool_name_info);
            }

            VkDescriptorSetLayoutBinding const buffer_descriptor_set_layout_binding{
                .binding = DAXA_STORAGE_BUFFER_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = static_cast<u32>(buffer_slots.max_resources),
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };

            VkDescriptorSetLayoutBinding const storage_image_descriptor_set_layout_binding{
                .binding = DAXA_STORAGE_IMAGE_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = static_cast<u32>(image_slots.max_resources),
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };

            VkDescriptorSetLayoutBinding const sampled_image_descriptor_set_layout_binding{
                .binding = DAXA_SAMPLED_IMAGE_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = static_cast<u32>(image_slots.max_resources),
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };

            VkDescriptorSetLayoutBinding const sampler_descriptor_set_layout_binding{
                .binding = DAXA_SAMPLER_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .descriptorCount = static_cast<u32>(sampler_slots.max_resources),
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };

            VkDescriptorSetLayoutBinding const buffer_address_buffer_descriptor_set_layout_binding{
                .binding = DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };

            auto descriptor_set_layout_bindings = std::vector{
                buffer_descriptor_set_layout_binding,
                storage_image_descriptor_set_layout_binding,
                sampled_image_descriptor_set_layout_binding,
                sampler_descriptor_set_layout_binding,
                buffer_address_buffer_descriptor_set_layout_binding,
            };

            if (ray_tracing_enabled)
            {
                VkDescriptorSetLayoutBinding const as_descriptor_set_layout_binding{
                    .binding = DAXA_ACCELERATION_STRUCTURE_BINDING,
                    .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    .descriptorCount = static_cast<u32>(tlas_slots.max_resources),
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .pImmutableSamplers = nullptr,
                };
                descriptor_set_layout_bindings.push_back(as_descriptor_set_layout_binding);
            }

            auto vk_descriptor_binding_flags = std::vector{
                VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
                VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
                VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
                VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
                VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
            };
            if (ray_tracing_enabled)
            {
                vk_descriptor_binding_flags.push_back({VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT});
            }

            VkDescriptorSetLayoutBindingFlagsCreateInfo vk_descriptor_set_layout_binding_flags_create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .pNext = nullptr,
                .bindingCount = static_cast<u32>(vk_descriptor_binding_flags.size()),
                .pBindingFlags = vk_descriptor_binding_flags.data(),
            };

            VkDescriptorSetLayoutCreateInfo const vk_descriptor_set_layout_create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &vk_descriptor_set_layout_binding_flags_create_info,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = static_cast<u32>(descriptor_set_layout_bindings.size()),
                .pBindings = descriptor_set_layout_bindings.data(),
            };

            result = static_cast<daxa_Result>(vkCreateDescriptorSetLayout(device, &vk_descriptor_set_layout_create_info, nullptr, &this->vk_descriptor_set_layout));
            _DAXA_RETURN_IF_ERROR(result, result)

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto const * name = "mega descriptor set layout";
                VkDebugUtilsObjectNameInfoEXT const name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_set_layout),
                    .pObjectName = name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &name_info);
            }

            VkDescriptorSetAllocateInfo const vk_descriptor_set_allocate_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = this->vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &this->vk_descriptor_set_layout,
            };

            result = static_cast<daxa_Result>(vkAllocateDescriptorSets(device, &vk_descriptor_set_allocate_info, &this->vk_descriptor_set));
            _DAXA_RETURN_IF_ERROR(result, result)

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto const * name = "mega descriptor set";
                VkDebugUtilsObjectNameInfoEXT const name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET,
                    .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_set),
                    .pObjectName = name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &name_info);
            }

            auto vk_descriptor_set_layouts = std::array{this->vk_descriptor_set_layout};
            VkPipelineLayoutCreateInfo vk_pipeline_create_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .setLayoutCount = static_cast<u32>(vk_descriptor_set_layouts.size()),
                .pSetLayouts = vk_descriptor_set_layouts.data(),
                .pushConstantRangeCount = 0,
                .pPushConstantRanges = nullptr,
            };

            result = static_cast<daxa_Result>(vkCreatePipelineLayout(device, &vk_pipeline_create_info, nullptr, pipeline_layouts.data()));
            _DAXA_RETURN_IF_ERROR(result, result)

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto const * name = "pipeline layout (push constant size 0)";
                VkDebugUtilsObjectNameInfoEXT const name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    .objectHandle = std::bit_cast<uint64_t>(*pipeline_layouts.data()),
                    .pObjectName = name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &name_info);
            }

            for (usize i = 1; i < DAXA_PIPELINE_LAYOUT_COUNT; ++i)
            {
                VkPushConstantRange const vk_push_constant_range{
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .offset = 0,
                    .size = static_cast<u32>(i * 4),
                };
                vk_pipeline_create_info.pushConstantRangeCount = 1;
                vk_pipeline_create_info.pPushConstantRanges = &vk_push_constant_range;
                result = static_cast<daxa_Result>(vkCreatePipelineLayout(device, &vk_pipeline_create_info, nullptr, &pipeline_layouts.at(i)));
                _DAXA_RETURN_IF_ERROR(result, result)

                if (vkSetDebugUtilsObjectNameEXT != nullptr)
                {
                    auto name = std::format("pipeline layout (push constant size {})", i * 4);
                    VkDebugUtilsObjectNameInfoEXT const name_info{
                        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                        .pNext = nullptr,
                        .objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                        .objectHandle = std::bit_cast<uint64_t>(pipeline_layouts.at(i)),
                        .pObjectName = name.c_str(),
                    };
                    vkSetDebugUtilsObjectNameEXT(device, &name_info);
                }
            }

            VkDescriptorBufferInfo const write_buffer{
                .buffer = device_address_buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            };

            VkWriteDescriptorSet const write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = this->vk_descriptor_set,
                .dstBinding = DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &write_buffer,
                .pTexelBufferView = nullptr,
            };

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
        else
        {
            const DescriptorHeapProperties& descriptor_heap_properties = descriptor_heap_properties_opt.value();
            heap_layout.storage_buffer_descriptor_size = descriptor_heap_properties.buffer_descriptor_size;
            heap_layout.image_descriptor_size = descriptor_heap_properties.image_descriptor_size;
            heap_layout.sampler_descriptor_size = descriptor_heap_properties.sampler_descriptor_size;
            heap_layout.acceleration_structure_descriptor_size = ray_tracing_enabled && vkGetPhysicalDeviceDescriptorSizeEXT != nullptr
                ? vkGetPhysicalDeviceDescriptorSizeEXT(physical_device, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                : 0u;

            heap_layout.storage_buffer_stride = align_up(heap_layout.storage_buffer_descriptor_size, descriptor_heap_properties.buffer_descriptor_alignment);
            heap_layout.image_stride = align_up(heap_layout.image_descriptor_size, descriptor_heap_properties.image_descriptor_alignment);
            heap_layout.sampler_stride = align_up(heap_layout.sampler_descriptor_size, descriptor_heap_properties.sampler_descriptor_alignment);
            heap_layout.acceleration_structure_stride = ray_tracing_enabled
                ? align_up(heap_layout.acceleration_structure_descriptor_size, descriptor_heap_properties.buffer_descriptor_alignment)
                : 0u;

            heap_layout.resource_heap_reserved_offset = 0;
            heap_layout.resource_heap_reserved_size = align_up(
                descriptor_heap_properties.min_resource_heap_reserved_range,
                descriptor_heap_properties.resource_heap_alignment);
            heap_layout.sampler_heap_reserved_offset = 0;
            heap_layout.sampler_heap_reserved_size = align_up(
                descriptor_heap_properties.min_sampler_heap_reserved_range,
                descriptor_heap_properties.sampler_heap_alignment);

            VkDeviceSize resource_cursor = align_up(heap_layout.resource_heap_reserved_size, descriptor_heap_properties.resource_heap_alignment);
            resource_cursor = align_up(resource_cursor, descriptor_heap_properties.buffer_descriptor_alignment);
            heap_layout.storage_buffer_offset = resource_cursor;
            resource_cursor += heap_layout.storage_buffer_stride * buffer_slots.max_resources;

            resource_cursor = align_up(resource_cursor, descriptor_heap_properties.image_descriptor_alignment);
            heap_layout.storage_image_offset = resource_cursor;
            resource_cursor += heap_layout.image_stride * image_slots.max_resources;

            resource_cursor = align_up(resource_cursor, descriptor_heap_properties.image_descriptor_alignment);
            heap_layout.sampled_image_offset = resource_cursor;
            resource_cursor += heap_layout.image_stride * image_slots.max_resources;

            resource_cursor = align_up(resource_cursor, descriptor_heap_properties.buffer_descriptor_alignment);
            heap_layout.buffer_device_address_offset = resource_cursor;
            resource_cursor += heap_layout.storage_buffer_stride;

            if (ray_tracing_enabled)
            {
                resource_cursor = align_up(resource_cursor, descriptor_heap_properties.buffer_descriptor_alignment);
                heap_layout.acceleration_structure_offset = resource_cursor;
                resource_cursor += heap_layout.acceleration_structure_stride * tlas_slots.max_resources;
            }

            heap_layout.resource_heap_size = align_up(resource_cursor, descriptor_heap_properties.resource_heap_alignment);

            VkDeviceSize sampler_cursor = align_up(heap_layout.sampler_heap_reserved_size, descriptor_heap_properties.sampler_heap_alignment);
            sampler_cursor = align_up(sampler_cursor, descriptor_heap_properties.sampler_descriptor_alignment);
            heap_layout.sampler_offset = sampler_cursor;
            sampler_cursor += heap_layout.sampler_stride * sampler_slots.max_resources;
            heap_layout.sampler_heap_size = align_up(sampler_cursor, descriptor_heap_properties.sampler_heap_alignment);

            auto fits_u32 = [](VkDeviceSize value) -> bool
            {
                return value <= static_cast<VkDeviceSize>(std::numeric_limits<u32>::max());
            };

            if (heap_layout.resource_heap_size > descriptor_heap_properties.max_resource_heap_size ||
                heap_layout.sampler_heap_size > descriptor_heap_properties.max_sampler_heap_size ||
                !fits_u32(heap_layout.storage_buffer_offset) ||
                !fits_u32(heap_layout.storage_buffer_stride) ||
                !fits_u32(heap_layout.storage_image_offset) ||
                !fits_u32(heap_layout.sampled_image_offset) ||
                !fits_u32(heap_layout.image_stride) ||
                !fits_u32(heap_layout.sampler_offset) ||
                !fits_u32(heap_layout.sampler_stride) ||
                !fits_u32(heap_layout.buffer_device_address_offset) ||
                (ray_tracing_enabled && (!fits_u32(heap_layout.acceleration_structure_offset) || !fits_u32(heap_layout.acceleration_structure_stride))))
            {
                // return DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_DESCRIPTOR_HEAP_SIZE;
            }

            VkBufferUsageFlags const heap_usage =
                VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

            VkBufferCreateInfo const resource_heap_buffer_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .size = heap_layout.resource_heap_size,
                .usage = heap_usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            };

            VmaAllocationCreateInfo const heap_allocation_info{
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .requiredFlags = {},
                .preferredFlags = {},
                .memoryTypeBits = std::numeric_limits<u32>::max(),
                .pool = nullptr,
                .pUserData = nullptr,
                .priority = 0.5f,
            };

            VmaAllocationInfo resource_heap_alloc_info = {};
            result = static_cast<daxa_Result>(vmaCreateBuffer(
                vma_allocator,
                &resource_heap_buffer_info,
                &heap_allocation_info,
                &resource_heap.vk_buffer,
                &resource_heap.vma_allocation,
                &resource_heap_alloc_info));
            _DAXA_RETURN_IF_ERROR(result, result);
            resource_heap.host_ptr = resource_heap_alloc_info.pMappedData;

            VkBufferCreateInfo const sampler_heap_buffer_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = {},
                .size = heap_layout.sampler_heap_size,
                .usage = heap_usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            };

            VmaAllocationInfo sampler_heap_alloc_info = {};
            result = static_cast<daxa_Result>(vmaCreateBuffer(
                vma_allocator,
                &sampler_heap_buffer_info,
                &heap_allocation_info,
                &sampler_heap.vk_buffer,
                &sampler_heap.vma_allocation,
                &sampler_heap_alloc_info));
            _DAXA_RETURN_IF_ERROR(result, result);
            sampler_heap.host_ptr = sampler_heap_alloc_info.pMappedData;

            VkBufferDeviceAddressInfo const resource_heap_address_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = resource_heap.vk_buffer,
            };
            resource_heap.device_address = vkGetBufferDeviceAddress(device, &resource_heap_address_info);
            resource_heap.size = heap_layout.resource_heap_size;

            VkBufferDeviceAddressInfo const sampler_heap_address_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = sampler_heap.vk_buffer,
            };
            sampler_heap.device_address = vkGetBufferDeviceAddress(device, &sampler_heap_address_info);
            sampler_heap.size = heap_layout.sampler_heap_size;

            resource_heap.bind_info = {
                .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                .pNext = nullptr,
                .heapRange = VkDeviceAddressRangeEXT{
                    .address = resource_heap.device_address,
                    .size = resource_heap.size,
                },
                .reservedRangeOffset = heap_layout.resource_heap_reserved_offset,
                .reservedRangeSize = heap_layout.resource_heap_reserved_size,
            };

            sampler_heap.bind_info = {
                .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                .pNext = nullptr,
                .heapRange = VkDeviceAddressRangeEXT{
                    .address = sampler_heap.device_address,
                    .size = sampler_heap.size,
                },
                .reservedRangeOffset = heap_layout.sampler_heap_reserved_offset,
                .reservedRangeSize = heap_layout.sampler_heap_reserved_size,
            };

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto const * resource_heap_name = "daxa resource descriptor heap";
                VkDebugUtilsObjectNameInfoEXT const resource_heap_name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_BUFFER,
                    .objectHandle = std::bit_cast<uint64_t>(resource_heap.vk_buffer),
                    .pObjectName = resource_heap_name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &resource_heap_name_info);

                auto const * sampler_heap_name = "daxa sampler descriptor heap";
                VkDebugUtilsObjectNameInfoEXT const sampler_heap_name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_BUFFER,
                    .objectHandle = std::bit_cast<uint64_t>(sampler_heap.vk_buffer),
                    .pObjectName = sampler_heap_name,
                };
                vkSetDebugUtilsObjectNameEXT(device, &sampler_heap_name_info);
            }

            descriptor_mappings = {};
            descriptor_mappings[0] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_STORAGE_BUFFER_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.storage_buffer_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.storage_buffer_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = 0,
                        .samplerHeapArrayStride = 0,
                    },
                },
            };
            descriptor_mappings[1] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_STORAGE_IMAGE_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.storage_image_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.image_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = 0,
                        .samplerHeapArrayStride = 0,
                    },
                },
            };
            descriptor_mappings[2] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_SAMPLED_IMAGE_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.sampled_image_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.image_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = 0,
                        .samplerHeapArrayStride = 0,
                    },
                },
            };
            descriptor_mappings[3] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_SAMPLER_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.sampler_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.sampler_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = static_cast<u32>(heap_layout.sampler_offset),
                        .samplerHeapArrayStride = static_cast<u32>(heap_layout.sampler_stride),
                    },
                },
            };
            descriptor_mappings[4] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.buffer_device_address_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.storage_buffer_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = 0,
                        .samplerHeapArrayStride = 0,
                    },
                },
            };
            descriptor_mappings[5] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_ACCELERATION_STRUCTURE_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{
                    .constantOffset = VkDescriptorMappingSourceConstantOffsetEXT{
                        .heapOffset = static_cast<u32>(heap_layout.acceleration_structure_offset),
                        .heapArrayStride = static_cast<u32>(heap_layout.acceleration_structure_stride),
                        .pEmbeddedSampler = nullptr,
                        .samplerHeapOffset = 0,
                        .samplerHeapArrayStride = 0,
                    },
                },
            };
            descriptor_mappings[6] = VkDescriptorSetAndBindingMappingEXT{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .pNext = nullptr,
                .descriptorSet = DAXA_GPU_TABLE_SET_BINDING,
                .firstBinding = DAXA_PUSH_DATA_BINDING,
                .bindingCount = 1,
                .resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_DATA_EXT,
                .sourceData = VkDescriptorMappingSourceDataEXT{.pushDataOffset = 0},
            };
        }

        return result;
    }

    void GPUShaderResourceTable::cleanup(VkDevice device, VmaAllocator vma_allocator)
    {
        [[maybe_unused]] auto print_remaining = [&](std::string prefix, auto & sro)
        {
            std::string ret{prefix + "\nthis can happen due to not waiting for the gpu to finish executing, as daxa defers destruction. List of survivors:\n"};
            for (u32 page_i = 0; page_i < sro.valid_page_count.load(); ++page_i)
            {
                auto & page = sro.paged_data[page_i];
                if (page)
                {
                    for (u32 i = 0; i < GpuResourcePool<>::PAGE_SIZE; ++i)
                    {
                        auto & slot = (*page.get())[i];
                        u32 const resource_i = page_i * GpuResourcePool<>::PAGE_SIZE + i;
                        bool non_zero_refcount = {};
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(slot)>, ImplBufferSlot>)
                        {
                            non_zero_refcount = GpuResourcePool<>::get_refcnt(sro.version_refcnt_of_slot(resource_i));
                        }
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(slot)>, ImplBlasSlot>)
                        {
                            non_zero_refcount = GpuResourcePool<>::get_refcnt(sro.version_refcnt_of_slot(resource_i));
                        }
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(slot)>, ImplTlasSlot>)
                        {
                            non_zero_refcount = GpuResourcePool<>::get_refcnt(sro.version_refcnt_of_slot(resource_i));
                        }
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(slot)>, ImplImageSlot>)
                        {
                            non_zero_refcount = GpuResourcePool<>::get_refcnt(sro.version_refcnt_of_slot(resource_i));
                        }
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(slot)>, ImplSamplerSlot>)
                        {
                            non_zero_refcount = GpuResourcePool<>::get_refcnt(sro.version_refcnt_of_slot(resource_i));
                        }
                        if (non_zero_refcount)
                        {
                            ret += std::format("debug name : \"{}\"", r_cast<SmallString const *>(&slot.info.name)->view());
                            ret += "\n";
                        }
                    }
                }
            }
            return ret;
        };
        DAXA_DBG_ASSERT_TRUE_M(buffer_slots.free_index_stack.size() == buffer_slots.next_index, print_remaining("Detected leaked buffers; not all buffers have been destroyed before destroying the device;", buffer_slots));
        DAXA_DBG_ASSERT_TRUE_M(image_slots.free_index_stack.size() == image_slots.next_index, print_remaining("Detected leaked images; not all images have been destroyed before destroying the device;", image_slots));
        DAXA_DBG_ASSERT_TRUE_M(sampler_slots.free_index_stack.size() == sampler_slots.next_index, print_remaining("Detected leaked samplers; not all samplers have been destroyed before destroying the device;", sampler_slots));

        if(resource_heap.vk_buffer == VK_NULL_HANDLE)
        {
            for (usize i = 0; i < DAXA_PIPELINE_LAYOUT_COUNT; ++i)
            {
                vkDestroyPipelineLayout(device, pipeline_layouts.at(i), nullptr);
            }
            vkDestroyDescriptorSetLayout(device, this->vk_descriptor_set_layout, nullptr);
            vkResetDescriptorPool(device, this->vk_descriptor_pool, {});
            vkDestroyDescriptorPool(device, this->vk_descriptor_pool, nullptr);
        }
        else
        {
            vmaDestroyBuffer(vma_allocator, resource_heap.vk_buffer, resource_heap.vma_allocation);
            vmaDestroyBuffer(vma_allocator, sampler_heap.vk_buffer, sampler_heap.vma_allocation);
        }
    }

    void write_descriptor_set_sampler(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkSampler vk_sampler, u32 index)
    {
        VkDescriptorImageInfo const vk_descriptor_image_info{
            .sampler = vk_sampler,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set_storage{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_SAMPLER_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = &vk_descriptor_image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set_storage, 0, nullptr);
    }

    void write_descriptor_heap_sampler(VkDevice vk_device, PFN_vkWriteSamplerDescriptorsEXT vkWriteSamplerDescriptorsEXT, GPUShaderResourceTable const & table, VkSamplerCreateInfo const & vk_sampler_create_info, u32 index)
    {
        auto * base = static_cast<std::byte *>(table.sampler_heap.host_ptr);
        VkDeviceSize const dst_offset = table.heap_layout.sampler_offset + table.heap_layout.sampler_stride * index;
        VkHostAddressRangeEXT const dst_range{
            .address = base + dst_offset,
            .size = static_cast<size_t>(table.heap_layout.sampler_descriptor_size),
        };
        vkWriteSamplerDescriptorsEXT(vk_device, 1, &vk_sampler_create_info, &dst_range);
    }

    void write_descriptor_set_buffer(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkBuffer vk_buffer, VkDeviceSize offset, VkDeviceSize range, u32 index)
    {
        VkDescriptorBufferInfo const vk_descriptor_image_info{
            .buffer = vk_buffer,
            .offset = offset,
            .range = range,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_STORAGE_BUFFER_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &vk_descriptor_image_info,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set, 0, nullptr);
    }

    void write_descriptor_heap_buffer(VkDevice vk_device, PFN_vkWriteResourceDescriptorsEXT vkWriteResourceDescriptorsEXT, GPUShaderResourceTable const & table, VkDeviceAddress address, VkDeviceSize range, u32 index)
    {
        auto * base = static_cast<std::byte *>(table.resource_heap.host_ptr);
        VkDeviceSize const dst_offset = table.heap_layout.storage_buffer_offset + table.heap_layout.storage_buffer_stride * index;
        VkHostAddressRangeEXT const dst_range{
            .address = base + dst_offset,
            .size = static_cast<size_t>(table.heap_layout.storage_buffer_descriptor_size),
        };
        VkDeviceAddressRangeEXT const address_range{
            .address = address,
            .size = range,
        };
        VkResourceDescriptorInfoEXT const resource_info{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .pNext = nullptr,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .data = VkResourceDescriptorDataEXT{.pAddressRange = &address_range},
        };
        vkWriteResourceDescriptorsEXT(vk_device, 1, &resource_info, &dst_range);
    }

    void write_descriptor_set_image(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkImageView vk_image_view, ImageUsageFlags usage, u32 index)
    {
        u32 descriptor_set_write_count = 0;
        std::array<VkWriteDescriptorSet, 2> descriptor_set_writes = {};

        VkDescriptorImageInfo const vk_descriptor_image_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = vk_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_STORAGE_IMAGE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &vk_descriptor_image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        if ((usage & ImageUsageFlagBits::SHADER_STORAGE) != ImageUsageFlagBits::NONE)
        {
            descriptor_set_writes.at(descriptor_set_write_count++) = vk_write_descriptor_set;
        }

        VkDescriptorImageInfo const vk_descriptor_image_info_sampled{
            .sampler = VK_NULL_HANDLE,
            .imageView = vk_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set_sampled{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_SAMPLED_IMAGE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &vk_descriptor_image_info_sampled,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        if ((usage & ImageUsageFlagBits::SHADER_SAMPLED) != ImageUsageFlagBits::NONE)
        {
            descriptor_set_writes.at(descriptor_set_write_count++) = vk_write_descriptor_set_sampled;
        }

        vkUpdateDescriptorSets(vk_device, descriptor_set_write_count, descriptor_set_writes.data(), 0, nullptr);
    }

    void write_descriptor_heap_image(VkDevice vk_device, PFN_vkWriteResourceDescriptorsEXT vkWriteResourceDescriptorsEXT, GPUShaderResourceTable const & table, VkImageViewCreateInfo const & vk_image_view_create_info, ImageUsageFlags usage, u32 index)
    {
        auto * base = static_cast<std::byte *>(table.resource_heap.host_ptr);
        VkImageDescriptorInfoEXT const image_descriptor_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
            .pNext = nullptr,
            .pView = &vk_image_view_create_info,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };

        if ((usage & ImageUsageFlagBits::SHADER_STORAGE) != ImageUsageFlagBits::NONE)
        {
            VkDeviceSize const dst_offset = table.heap_layout.storage_image_offset + table.heap_layout.image_stride * index;
            VkHostAddressRangeEXT const dst_range{
                .address = base + dst_offset,
                .size = static_cast<size_t>(table.heap_layout.image_descriptor_size),
            };
            VkResourceDescriptorInfoEXT const resource_info{
                .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                .pNext = nullptr,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .data = VkResourceDescriptorDataEXT{.pImage = &image_descriptor_info},
            };
            vkWriteResourceDescriptorsEXT(vk_device, 1, &resource_info, &dst_range);
        }

        if ((usage & ImageUsageFlagBits::SHADER_SAMPLED) != ImageUsageFlagBits::NONE)
        {
            VkDeviceSize const dst_offset = table.heap_layout.sampled_image_offset + table.heap_layout.image_stride * index;
            VkHostAddressRangeEXT const dst_range{
                .address = base + dst_offset,
                .size = static_cast<size_t>(table.heap_layout.image_descriptor_size),
            };
            VkResourceDescriptorInfoEXT const resource_info{
                .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
                .pNext = nullptr,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .data = VkResourceDescriptorDataEXT{.pImage = &image_descriptor_info},
            };
            vkWriteResourceDescriptorsEXT(vk_device, 1, &resource_info, &dst_range);
        }
    }

    void write_descriptor_set_acceleration_structure(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkAccelerationStructureKHR vk_acceleration_structure, u32 index)
    {
        VkWriteDescriptorSetAccelerationStructureKHR vk_write_descriptor_set_as = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &vk_acceleration_structure,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &vk_write_descriptor_set_as,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_ACCELERATION_STRUCTURE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set, 0, nullptr);
    }

    void write_descriptor_heap_acceleration_structure(VkDevice vk_device, PFN_vkWriteResourceDescriptorsEXT vkWriteResourceDescriptorsEXT, GPUShaderResourceTable const & table, VkDeviceAddress address, u32 index)
    {
        auto * base = static_cast<std::byte *>(table.resource_heap.host_ptr);
        VkDeviceSize const dst_offset = table.heap_layout.acceleration_structure_offset + table.heap_layout.acceleration_structure_stride * index;
        VkHostAddressRangeEXT const dst_range{
            .address = base + dst_offset,
            .size = static_cast<size_t>(table.heap_layout.acceleration_structure_descriptor_size),
        };
        VkDeviceAddressRangeEXT const address_range{
            .address = address,
            .size = VK_WHOLE_SIZE,
        };
        VkResourceDescriptorInfoEXT const resource_info{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .pNext = nullptr,
            .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .data = VkResourceDescriptorDataEXT{
                .pAddressRange = &address_range
            },
        };
        vkWriteResourceDescriptorsEXT(vk_device, 1, &resource_info, &dst_range);
    }
} // namespace daxa
