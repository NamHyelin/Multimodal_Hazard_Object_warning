#include "jpeg_stream_server.hpp"
#include <iostream>

int main()
{
  const std::string name = "transport_test_" + std::to_string(::getpid());
  JpegStreamServer server(name);
  std::cout << "perception_" << name << "_jpeg_v1" << std::endl;
  std::uint32_t sec, nsec, size;
  while (std::cin >> sec >> nsec >> size) {
    if (size < 4 || size > 2000000) {return 2;}
    std::vector<std::uint8_t> jpeg(size, 7);
    jpeg[0] = 255; jpeg[1] = 216; jpeg[size - 2] = 255; jpeg[size - 1] = 217;
    server.publish(jpeg, sec, nsec);
    std::cout << "published" << std::endl;
  }
  server.stop();
}
