
__device__ void HSV2RGB(float h, float s, float v, float &r, float &g, float &b) {
		if(h < 0) {
			r=v;
			g=v;
			b=v;
			return;
		}
		h *= .0166666666666667; // convert from 360 to 0-6;		
		int i = (int) floor(h);
		float f = h - i;
		f = (!(i&1)) ?  1-f : f; // if even
		float m = v * (1-s); 
		float n = v * (1-s * f); 
		switch(i) {
			case 6:
			case 0:
				r = v;
				g= n;
				b = m;
				return;
			case 1:
				r = n;
				g= v;
				b = m;
				return;
			case 2:
				r = m;
				g= v;
				b = n;
				return;
			case 3:
				r = m;
				g= n;
				b = v;
				return;

			case 4:
				r = n;
				g= m;
				b = v;
				return;
			case 5:
				r = v;
				g= n;
				b = m;
				return;
		}

	}



__global__ void gpu_calcColor_kernel(int pointCnt, float* pixels, float minZ, float diffZ, float minC1, float minC2, float minC3, float diffC1, float diffC2, float diffC3, bool hsv, bool quads, float* colors) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;	
	
	
	bool validPoint = ((i > 0) && (i < pointCnt))? true : false;
	i *=3;	
	float z = validPoint ? pixels[i+2] : 0.0;
	if(z != 0.0) { // only equal to 0.0 if set as invalid in cloud contructor	
		float distZ = z - minZ;
		float percent = distZ/diffZ;
		percent = percent<=0.0 ? 0.01 : percent;
		percent = percent>=1.0 ? 1.0 : percent;
	
		float r; 	
		float g; 
		float b;
		if(hsv) {
			if(percent == 0.0) {
				HSV2RGB(minC1 , minC2, minC3 , r,g,b); 		
				
			} else {
				HSV2RGB(minC1 + percent * diffC1, minC2 + percent * diffC2, minC3 + percent * diffC3, r,g,b); 		
			}
		} else {
			r= minC1 + percent * diffC1;
			g= minC2 + percent * diffC2;
			b= minC3 + percent * diffC3;
		}
		
		if(quads) {
			i*=4;
			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i] = b;
			
		} else {
			colors[i++] = r;
			colors[i++] = g;
			colors[i] = b;
		}



	}

} 

extern "C" void gpu_calcColor(int pointCnt, float* pixels, float minZ, float diffZ, float minC1, float minC2, float minC3, float diffC1, float diffC2, float diffC3, bool hsv, bool quads, float* colors)
{
	int theadsPerBlock = 256;
	int blocks = (int) ceilf(pointCnt/(float) theadsPerBlock);
	gpu_calcColor_kernel <<<blocks,theadsPerBlock>>> (pointCnt, pixels, minZ, diffZ, minC1, minC2, minC3, diffC1, diffC2, diffC3, hsv, quads, colors);

};

