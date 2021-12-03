#include "CommandList.hpp"
#include "common.hpp"

namespace daxa {
	namespace gpu {

		DAXA_DEFINE_TRIVIAL_MOVE(CommandList)

		CommandList::CommandList() {
			this->renderAttachmentBuffer.reserve(10);
			usedBuffers.reserve(10);
			usedGraphicsPipelines.reserve(10);
			usedImages.reserve(10);
		}

		CommandList::~CommandList() {
			if (device) {
				assert(operationsInProgress == 0);
				assert(empty);
				vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
				vkDestroyCommandPool(device, cmdPool, nullptr);
				printf("command list complete destruction!\n");
				std::memset(this, 0, sizeof(CommandList));
			}
		}

		void CommandList::begin() {
			operationsInProgress += 1;
			VkCommandBufferBeginInfo cbbi{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.pNext = nullptr,
				.flags = VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};
			vkBeginCommandBuffer(cmd, &cbbi);
		}

		void CommandList::end() {
			operationsInProgress -= 1;
			vkEndCommandBuffer(cmd);
		}

		void CommandList::beginRendering(BeginRenderingInfo ri) {
			operationsInProgress += 1;
			for (int i = 0; i < ri.colorAttachments.size(); i++) {
				usedImages.push_back(ri.colorAttachments[i].image);

				renderAttachmentBuffer.push_back(VkRenderingAttachmentInfoKHR{
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
					.pNext = nullptr,
					.imageView = ri.colorAttachments[i].image->getVkView(),
					.imageLayout = ri.colorAttachments[i].image->getLayout(),
					.resolveMode = VK_RESOLVE_MODE_NONE,// ri.colorAttachments[i].resolveMode,
					.loadOp = ri.colorAttachments[i].loadOp,
					.storeOp = ri.colorAttachments[i].storeOp,
					.clearValue = ri.colorAttachments[i].clearValue
				});
			}

			std::optional<VkRenderingAttachmentInfoKHR> depthAttachmentInfo = ri.depthAttachment == nullptr ? std::nullopt :
				std::optional{
					VkRenderingAttachmentInfoKHR{
						.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
						.pNext = nullptr,
						.imageView = ri.depthAttachment->image->getVkView(),
						.imageLayout = ri.depthAttachment->image->getLayout(),
						.resolveMode = ri.depthAttachment->resolveMode,
						.loadOp = ri.depthAttachment->loadOp,
						.storeOp = ri.depthAttachment->storeOp
					}
				};

			std::optional<VkRenderingAttachmentInfoKHR> stencilAttachmentInfo = ri.stencilAttachment == nullptr ? std::nullopt :
				std::optional{
					VkRenderingAttachmentInfoKHR{
						.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
						.pNext = nullptr,
						.imageView = ri.stencilAttachment->image->getVkView(),
						.imageLayout = ri.stencilAttachment->image->getLayout(),
						.resolveMode = ri.stencilAttachment->resolveMode,
						.loadOp = ri.stencilAttachment->loadOp,
						.storeOp = ri.stencilAttachment->storeOp
					}
				};

			VkRenderingInfoKHR renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
			renderInfo.pNext = nullptr;
			if (ri.renderArea) {
				renderInfo.renderArea = *ri.renderArea;
			}
			else if (ri.colorAttachments.size() > 0) {
				renderInfo.renderArea.extent.width = ri.colorAttachments[0].image->getExtent().width;
				renderInfo.renderArea.extent.height = ri.colorAttachments[0].image->getExtent().height;
			}
			else if (ri.depthAttachment != nullptr) {
				renderInfo.renderArea.extent.width = ri.depthAttachment->image->getExtent().width;
				renderInfo.renderArea.extent.height = ri.depthAttachment->image->getExtent().height;
			}
			else if (ri.stencilAttachment != nullptr) {
				renderInfo.renderArea.extent.width = ri.stencilAttachment->image->getExtent().width;
				renderInfo.renderArea.extent.height = ri.stencilAttachment->image->getExtent().height;
			}	// otherwise let it be zero, as we dont render anything anyways

			renderInfo.layerCount = 1;	// Not sure what this does

			renderInfo.colorAttachmentCount = ri.colorAttachments.size();
			renderInfo.pColorAttachments = renderAttachmentBuffer.data();
			renderInfo.pDepthAttachment = depthAttachmentInfo.has_value() ? &depthAttachmentInfo.value() : nullptr;
			renderInfo.pStencilAttachment = stencilAttachmentInfo.has_value() ? &stencilAttachmentInfo.value() : nullptr;

			this->vkCmdBeginRenderingKHR(cmd, (VkRenderingInfoKHR*)&renderInfo);

			renderAttachmentBuffer.clear();
		}
		void CommandList::endRendering() {

			operationsInProgress -= 1;
			this->vkCmdEndRenderingKHR(cmd);
		}

		void CommandList::bindPipeline(GraphicsPipelineHandle graphicsPipeline) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->getVkPipeline());
			usedGraphicsPipelines.push_back(graphicsPipeline);
		}

		void CommandList::reset() {
			assert(operationsInProgress == 0);
			empty = true;
			vkResetCommandPool(device, cmdPool, VkCommandPoolResetFlagBits::VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
			usedBuffers.clear();
			usedImages.clear();
			usedGraphicsPipelines.clear();
		}

		void CommandList::changeImageLayout(ImageHandle image, VkImageLayout newLayout) {

			VkImageMemoryBarrier imgMemBarr = {};
			imgMemBarr.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imgMemBarr.pNext = nullptr;
			imgMemBarr.oldLayout = image->getLayout();
			imgMemBarr.newLayout = newLayout;
			imgMemBarr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imgMemBarr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imgMemBarr.image = image->getVkImage();
			imgMemBarr.subresourceRange = VkImageSubresourceRange{
				.aspectMask = image->getVkAspect(),
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			};
			imgMemBarr.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
			imgMemBarr.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

			vkCmdPipelineBarrier(
				cmd,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_DEPENDENCY_DEVICE_GROUP_BIT,
				0,
				nullptr,
				0,
				nullptr,
				1,
				&imgMemBarr
			);

			image->layout = newLayout;
			usedImages.push_back(std::move(image));
		}

		void CommandList::setViewport(VkViewport const& viewport) {
			vkCmdSetViewport(cmd, 0, 1, &viewport);
		}

		void CommandList::setScissor(VkRect2D const& scissor) {
			vkCmdSetScissor(cmd, 0, 1, &scissor);
		}

		void CommandList::draw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) {
			vkCmdDraw(cmd, vertexCount, instanceCount, firstVertex, firstInstance);
		}

		void CommandList::copyBufferToBuffer(BufferHandle src, BufferHandle dst, std::span<VkBufferCopy> copyRegions) {
			assert(copyRegions.size() > 0, "ERROR: tried copying 0 regions from buffer to buffer, this is a bug!");
			for (int i = 0; i < copyRegions.size(); i++) {
				assert(src->getSize() >= copyRegions[i].size + copyRegions[i].srcOffset, "ERROR: src buffer is smaller than the region that shouly be copied!");
				assert(dst->getSize() >= copyRegions[i].size + copyRegions[i].dstOffset, "ERROR: dst buffer is smaller than the region that shouly be copied!");
			}
			vkCmdCopyBuffer(cmd, src->getVkBuffer(), dst->getVkBuffer(), copyRegions.size(), copyRegions.data());
		}
	}
}