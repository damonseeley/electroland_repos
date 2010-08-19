#include "PersonDetector.h"
#include "Globals.h"

/*
float PersonDetector::personFilter[] = {    
	-0.64f,	-0.64f,	-0.64f,	-0.64f,	-0.64f,
	-0.64f,	1.0f,	1.0f,	1.0f,	-0.64f,
	-0.64f,	1.0f,	1.0f,	1.0f,	-0.64f,
	-0.64f,	1.0f,	1.0f,	1.0f,	-0.64f,
	-0.64f,	-0.64f,	-0.64f,	-0.64f,	-0.64f
};
*/



PersonDetector::PersonDetector(Projection *proj) {

	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1,1);

	this->projection = proj;
	heights = new float[proj->div.x * proj->div.z];  
	conv = new float[proj->div.x * proj->div.z];  
	localMax = new float[proj->div.x * proj->div.z];  
	filter = new float[400];
	constructConv(filter, 20, 20);
//	 = cv::cvarrToMatCreateMat(proj->div.x, proj->div.y, CV_8UC1);
//		CvMat cvMatHeights;

	cvMatThresh = *cvCreateMat(proj->div.z, proj->div.x, CV_8UC1);
	cvMatCont = *cvCreateMat(proj->div.z, proj->div.x, CV_8UC1);
	cvMatDisplay = *cvCreateMat(proj->div.z * 5, proj->div.x * 5, CV_8UC1);
	cvMatContMask = *cvCreateMat(proj->div.z, proj->div.x, CV_8UC1);

	g_storage = cvCreateMemStorage(0);

	
	cvNamedWindow( "Contours", 1 );
	cvResizeWindow( "Contours", 100,100 );

	 minPersonArea = 50; // in pixels
	 maxPersonArea = 200; // in pixels
	 minHandDist = 10; // in pixels
	maxMatchDistSqr = 1; // in meters sqr
	curFrame = 0;
	 frameLife = 7;



}			

#define ECONST 2.71828183

float PersonDetector::guassian(int dx, int dy, float sig) {
	return (1.0/ (2.0 * 3.1415 *sig *sig)) * 
		pow( (double) ECONST, (double) -(dx*dx)/2*(sig*sig)); 
}

void PersonDetector::constructConv(float *a, int w, int h) {
	int cX = w * .5f;
	int cY = h * .5f;
	int sum = 0;
	for(int i = 0; i < w; i++) {
		for(int j = 0; j < h; j++) {
			int dx = i - cX;
			int dy = i - cY;
			dx = (dx <= 0) ? -dx : 0;
			dy = (dy <= 0) ? -dy : 0;
			int val = ((cX+1)-dx) + ((cY +1)-dy);
			val *= val;
			a[i + j * w] = val  ;
			sum += val;
		}
	}
	for(int i = 0; i < w*h;i++) {
		a[i]/=(float) sum;
	}

}
float PersonDetector::distSQR(CvPoint* p1, CvPoint* p2) {
	float dx = p1->x - p2->x;
	float dy = p1->y - p2->y;
	return (dx*dx)+(dy*dy);

}

void PersonDetector::calc(long curFrame) {
	newTracks.clear();
	projection->copyMaxTo(heights);
	
	cvInitMatHeader(&cvMatHeights, projection->div.z, projection->div.x, CV_32FC1, heights);


	cvThreshold(&cvMatHeights, &cvMatThresh, 15, 255, CV_THRESH_BINARY);

	CvSeq* contours = 0;

//	CvContourScanner cvStartFindContours
//	cvCopy(&cvMatThresh,&cvMatCont); // copy before finding contours

	cvFindContours(&cvMatThresh, g_storage, &contours, sizeof(CvContour));
	

//	CvSeq* contour = cvFindNextContour(contours);
//	int cnt = 0;
//	double areaSum = 0;
//	while(contour) {
//		cnt++;
//		double area = cvContourArea(contour);
//		areaSum +=area;
//
//	}
//	std::cout << "ave contour area " << area / cnt << std::endl;

	cvZero(&cvMatCont);
	
	curFrame++;
	long propGoodUntil = curFrame + frameLife;

//good explanion of moments
//	http://public.cranfield.ac.uk/c5354/teaching/dip/opencv/SimpleImageAnalysisbyMoments.pdf
	if( contours ){
		CvSeq* contour = contours;
		for( ; contour != 0; contour = contour->h_next ) {
			double area = cvContourArea(contour);
			if((area > minPersonArea) && (area < maxPersonArea)) {
				
				CvMoments moments;
				cvMoments( contour, &moments);
				float imgX = moments.m10/moments.m00; // center of contour
				float imgZ = moments.m01/moments.m00;

				cvZero(&cvMatContMask);
				// draw once for display and once for mask - is there a fater way?
				cvDrawContours( &cvMatCont, contour, cvScalarAll(100), cvScalarAll(125), -1, -1);
				cvDrawContours( &cvMatContMask, contour, cvScalarAll(255), cvScalarAll(255), -1, -1);

				CvScalar avgHeight = cvAvg(&cvMatHeights, &cvMatContMask);
				float imgY = avgHeight.val[0];
				/*				

				float imgY = 0;
				int cnt = 0;
				for(int pixi = -1; pixi < 2; pixi++) {
				for(int pixj = -1; pixj < 2; pixj++) {
					int pixX = imgX + pixi;
					int pixZ = imgZ + pixj;
					if(((pixX >=0) && (pixZ >= 0)) && ((pixZ < projection->div.z) && (pixX < projection->div.x))) {
						float pix = heights[ (int) imgX +(  (int) imgZ * projection->div.x)];
						if(pix > 0) {
							imgY += heights[ (int) imgX + 1 +(  (int) imgZ * projection->div.x)];
							cnt++;
						}
					}

				}
				}
				imgY /= cnt;
				*/

				CvPoint center = cvPoint(imgX,imgZ);
				cvDrawCircle(&cvMatCont, center, 2, cvScalarAll(255),1);

				Track *t = new Track();
				t->id = curTrackID++;
				t->lastUpdated = curFrame;
				// need more efficent conversion from img to world coords TODO
				t->x->updateValue(((imgX/projection->div.x) * (projection->maxLoc.x - projection->minLoc.x)) + projection->minLoc.x, propGoodUntil);
				t->z->updateValue(((imgZ/projection->div.z) * (projection->maxLoc.z - projection->minLoc.z)) + projection->minLoc.z, propGoodUntil);

				float curCenter = ((imgY/projection->div.y)*(projection->maxLoc.y - projection->minLoc.y)) + projection->minLoc.y;
				t->center->updateValue(curCenter , propGoodUntil);

				float distsqr = minHandDist * minHandDist;
				CvPoint* maxPoint= NULL;
				//CvPoint* maxPoint= CV_GET_SEQ_ELEM( CvPoint, contour, 0 );
				//distSQR(maxPoint, &center);
				for( int i=0; i< contour->total; ++i ){
					CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, contour, i );
					float dist = distSQR(p, &center);
					if(dist >distsqr) {
						distsqr = dist;
						maxPoint = p;
					}
				}
				if(maxPoint) {
					cvDrawCircle(&cvMatCont, *maxPoint, 1, cvScalarAll(255),1);

					float hy = heights[ (int) maxPoint->x+ (int)( maxPoint->y * projection->div.x)];

					t->lhX->updateValue(maxPoint->x,propGoodUntil);
					t->lhY->updateValue(hy,propGoodUntil);
					t->lhZ->updateValue(maxPoint->y,propGoodUntil);


				}
				newTracks.addTrack(t);
				
			}
		}

				
			
		
		
		
/*
		cvDrawContours(
			&cvMatCont,
			contours,
			cvScalarAll(100),
			cvScalarAll(255),
			2 );
*/
		
	}

	cvResize(&cvMatCont, &cvMatDisplay);

	existingTracks.merge(&newTracks,maxMatchDistSqr, curFrame);

//	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);




	//	cvShowImage( "Contours", g_gray );

	cvShowImage("Contours", &cvMatDisplay);


//	cvThreshold(&cvMatHeights, &cvMatThresh, 1.0, 1.0, CV_THRESH_BINARY );
//	cvClearMemStorage( g_storage );
//

//	cvThreshold( g_gray, g_gray, g_thresh, 255, CV_THRESH_BINARY );
//	cvMatThresh
//	vector<vector<cv::Point> > contours;
//	vector<cv::Vec4i> hierarchy;
//	cv::findContours( cvMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

//	cpu_local_max(heights, localMax, projection->div.x, projection->div.z, 20,20, 1.0);
	//	cpu_convole(heights, conv, projection->div.x, projection->div.z, personFilter, 5,5,false);
}

void PersonDetector::render(float *map) {
	if(map == NULL){
//		newTracks.render();

		existingTracks.render();
//		newTrackHash-
	} else {
	glBegin(GL_QUADS);
	float left = 0;
	float right = projection->cellRenderSize.x;
	for(int x = 0; x < projection->div.x; x++) {
		float back = 0;
		float front = projection->cellRenderSize.z;
		for(int z = 0; z < projection->div.z; z++) {
			float cell = map[x + (z * projection->div.x)];
			float totalP = (float) cell / (float) projection->div.z;

			glColor3f(totalP,totalP,totalP);
			glVertex3f(left, projection->maxLoc.y, back );
			glVertex3f(right,  projection->maxLoc.y, back);
			glVertex3f(right, projection->maxLoc.y, front);
			glVertex3f(left, projection->maxLoc.y, front);

			back = front;
			front += projection->cellRenderSize.z;
		}
		left = right;
		right+=projection->cellRenderSize.x;
	}
	glEnd();
	}

}

// maximum within neighborhood iff average val in neighborhood is greater than thresh
void PersonDetector::cpu_local_max(float* d_src, float* d_dst, int w, int h, int nHoodWidth, int nHoodHeight, float aveThresh) {
	for(int ind = 0; ind < w*h; ind++) {

		if( ind < (w*h)) {
			int y = ind /  w;
			int x = ind % w;

			float origCellVal = d_src[x + y*w];
			float localMax = 0;
			float sum = 0;

			int halfWidth = nHoodWidth/2;
			int halfHeight = nHoodHeight/2;

			int convX = 0;
			for(int i  = -halfWidth; i <= halfWidth; i++) {
				int convY = 0;
				for(int j  = -halfHeight; j <= halfHeight; j++) {
					int xOff = x + i;
					int yOff = y + j;
					if(((xOff>=0) && (yOff>=0))&&((xOff< w)&&(yOff< h))) {
						float cellVal =  d_src[xOff +  (yOff * w)];
						sum += cellVal;
						localMax = cellVal > localMax ? cellVal : localMax;
					}
					convY++;	
				}
				convX++;
			}
			if((sum / (float) (nHoodWidth * nHoodHeight) > aveThresh) && (origCellVal >= localMax)) {
				d_dst[x+y*w] = localMax;
			} else {
				d_dst[x+y*w] = 0;
			}
		}
	}


}

void PersonDetector::cpu_convole(float* d_src, float* d_dst, int w, int h, float *d_conv, int cWidth, int cHeight, bool mirrorBoarder) {
	for(int ind = 0; ind < w*h; ind++) {

		if( ind < (w*h)) {
			int y = ind /  w;
			int x = ind % w;

			float origCellVal = d_src[x + y*w];
			float sum = 0;

			int halfWidth = cWidth/2;
			int halfHeight = cHeight/2;
			int convX = 0;
			for(int i  = -halfWidth; i <= halfWidth; i++) {
				int convY = 0;
				for(int j  = -halfHeight; j <= halfHeight; j++) {
					int xOff = x + i;
					int yOff = y + j;
					float cellVal = 0;
					if(((xOff<0) || (yOff<0))||((xOff>=w)||(yOff>=h))) {
						if(mirrorBoarder)		
							cellVal = origCellVal;
					} else {
						cellVal = d_src[xOff +  (yOff * w)];
					}
					sum+= cellVal * d_conv[convX + (convY * cWidth)];
					convY++;	
				}
				convX++;
			}
			d_dst[x+y*w] = sum;
		}
	}


}
