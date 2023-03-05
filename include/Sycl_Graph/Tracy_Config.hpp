#ifndef SYCL_GRAPH_TRACY_CONFIG_HPP
#define SYCL_GRAPH_TRACY_CONFIG_HPP
#include <CL/sycl.hpp>
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped 
#define FrameMark
#define FrameMarkStart(name)
#define FrameMarkEnd(name)
#endif
#ifdef TRACY_OPENCL_ENABLE
#include <tracy/TracyOpenCL.hpp>
namespace Sycl_Graph::Sycl
{

    void kernel_submit(sycl::queue& q, auto kernel, std::string name = {})
    {
        auto event = q.submit(kernel);
        auto ctx = q.get_context().get();
        if (!name.empty())
        {
            TracyCLZone(ctx, name.c_str());
        }
        else
        {
            TracyCLZone(ctx);
        }

    }
}
#else
#define TracyCLContext(ctx, device)
#define TracyCLDstroy(ctx)
#define TracyCLContextName(ctx, name, size)
#define TracyCLZone(ctx, name)
#define TracyCLSetEvent(event)
#endif

#endif