package net.electroland.elvis.imaging;
import static com.googlecode.javacv.cpp.opencv_core.cvAbsDiff;

import com.googlecode.javacv.cpp.opencv_core.CvArr;

public class ImageDifference {



	public static void apply(CvArr a, CvArr b, CvArr dest) {
			cvAbsDiff(a,b, dest);	
		
	}





}

