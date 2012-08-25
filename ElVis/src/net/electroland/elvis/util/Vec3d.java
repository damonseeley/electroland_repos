package net.electroland.elvis.util;

import com.googlecode.javacv.cpp.opencv_core.CvPoint;

public class Vec3d {
	double x,y,z;
	public Vec3d(CvPoint p) {
		this(p.x(), p.y(), 0);
	}
	
	public Vec3d(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	public static Vec3d cross(Vec3d a, Vec3d b) {
		return new Vec3d(
				a.y* b.z - a.z*b.y,
				a.z* b.x - a.x*b.z,
				a.x* b.y - a.y*b.x
				);
	}
	
	public static Vec3d add(Vec3d a, Vec3d b) {
		return new Vec3d(
				a.x + b.x,
				a.y + b.y,
				a.z + b.z
				);
	}

	public static Vec3d sub(Vec3d a, Vec3d b) {
		return new Vec3d(
				a.x - b.x,
				a.y - b.y,
				a.z - b.z
				);
	}

}
