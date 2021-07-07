all:
	g++ kf.cpp -std=c++14 `pkg-config opencv4 --cflags` `pkg-config opencv4 --libs`
