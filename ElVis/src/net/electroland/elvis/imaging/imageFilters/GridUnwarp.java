package net.electroland.elvis.imaging.imageFilters;

import static com.googlecode.javacv.cpp.opencv_core.CV_32F;
import static com.googlecode.javacv.cpp.opencv_core.CV_64F;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvMatMul;
import static com.googlecode.javacv.cpp.opencv_core.cvScalar;
import static com.googlecode.javacv.cpp.opencv_core.cvZero;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvRemap;

import java.awt.Polygon;
import java.awt.geom.Point2D;
import java.util.Vector;

import net.electroland.elvis.blobktracking.core.GridDesigner;
import net.electroland.elvis.util.BilinearInterp;
import net.electroland.elvis.util.ElProps;
import net.electroland.elvis.util.parameters.BoolParameter;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class GridUnwarp extends Filter {

	public static final CvScalar BLACK = cvScalar(0,0,0,0);


	boolean mapNeedsUpdate = true;

	public static int IS_ON = 0;

	IplImage mapx= null;
	IplImage mapy;

	CvMat matx;
	CvMat maty; 

	int width;
	int height;

	CvPoint[][] grid;

	CvMat compute_projective_matrix(CvPoint A, CvPoint B, CvPoint C,CvPoint D) {
		double aux_a = B.x()-D.x();
		double aux_b = C.x()-D.x();
		double aux_c = D.x()+A.x()-B.x()-C.x();
		double aux_d = B.y()-D.y();
		double aux_e = C.y()-D.y();
		double aux_f = D.y()+A.y()-B.y()-C.y();
		double h = (aux_a*aux_f-aux_d*aux_c)/(aux_a*aux_e-aux_d*aux_b);
		double g = (aux_c-aux_b*h)/aux_a;

		CvMat result = CvMat.create(3, 3);
		result.put(0, 0, B.x()*(g+1.0)-A.x());
		result.put(0, 1, C.x()*(h+1.0)-A.x());
		result.put(0, 2, A.x());
		result.put(1, 0, B.y()*(g+1.0)-A.y());
		result.put(1, 1, C.y()*(h+1.0)-A.y());
		result.put(1, 2, A.y());
		result.put(2, 0, g);
		result.put(2, 1, h);
		result.put(2, 2, 1.0);
		return result;
	}


	public GridUnwarp(int width, int height, CvPoint[][] grid, ElProps props) {
		super();
		this.width = width;
		this.height = height;
		parameters.add(new BoolParameter("gridUnwarpIsOn", true, props));
		this.grid = grid;

	}

	public int getWidth() {
		return width;
	}

	public void updateGrid(CvPoint[][] grid) {
		this.grid = grid;
		mapNeedsUpdate = true;
	}
	public int getHeight() {
		return height;
	}
	public void createMap() {
		mapNeedsUpdate = false;
		if(parameters.get(IS_ON).getBoolValue()) {
			createMap(grid);
		} else {
			mapx = null;
		}
	}

	public boolean isInside(int x, int y, CvPoint ul, CvPoint ur, CvPoint lr, CvPoint ll) {
		Polygon p = new Polygon();
		p.addPoint(ul.x(), ul.y());
		p.addPoint(ur.x(), ur.y());
		p.addPoint(lr.x(), lr.y());
		p.addPoint(ll.x(), ll.y());
		return p.contains(x, y);
	}

	public static int tmpi = 0;
	public static int tmpj = 0;

	public CvPoint diff(CvPoint pt1, CvPoint pt2) {
		return new CvPoint(pt1.x() - pt2.x(), pt1.y() - pt2.y());
	}

	public CvPoint getGridLoc(int x, int y, CvPoint[][] grid) {
		for(int i = 0; i < grid.length-1; i++) {
			for(int j = 0; j < grid[0].length-1; j++) {
				CvPoint ul = grid[i][j];
				CvPoint ur = grid[i+1][j];
				CvPoint lr = grid[i+1][j+1];
				CvPoint ll = grid[i][j+1];
				if(isInside(x, y, ul, ur, lr, ll)) {
					return new CvPoint(i,j);
				}
			}

		}
		return null;


	}
	public void createMap(CvPoint[][] grid) {
		matx = CvMat.create(height, width, CV_32F);
		maty = CvMat.create(height, width, CV_32F);

		CvMat[][] transforms = new CvMat[grid.length-1][grid[0].length-1];

		CvPoint[][] orig = GridDesigner.makeGrid(grid.length-1, grid[0].length-1, width, height);

		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {

				CvPoint ij = this.getGridLoc(x, y, grid);
				if(ij != null) {
					int i = ij.x();
					int j = ij.y();
					CvPoint oul = orig[i][j];
					CvPoint our = orig[i+1][j];
					CvPoint oll = orig[i][j+1];
					CvPoint olr = orig[i+1][j+1];

					CvPoint nul = grid[i][j];
					CvPoint nur =  grid[i+1][j];
					CvPoint nll =  grid[i][j+1];
					CvPoint nlr = grid[i+1][j+1];


					//					QuadUVMapper mapperO = new QuadUVMapper(oul, our, oll, olr);
					//					QuadUVMapper mapperN = new QuadUVMapper(nul, nur, nll, nlr);

					Vector<Point2D> uv = BilinearInterp.inverseBilerp(nul, nur, nlr, nll, new CvPoint(x,y));
					if(uv.size() >= 1) {
						Point2D newPoint = BilinearInterp.bilerp(oul, our, olr, oll, uv.get(0));
						matx.put(y, x, newPoint.getX());
						maty.put(y, x, newPoint.getY());
						if(uv.size() > 1) {
							System.out.println("two solutions " + uv.get(0)+ "   " + uv.get(1));
						}
					} else {
						matx.put(y,x, -1);
						maty.put(y,x, -1);

					}

				} else {
					matx.put(y,x, -1);
					maty.put(y,x, -1);
				}

			}

		}



		mapy =  new IplImage (maty); 
		mapx =  new IplImage (matx); 


	}


	public void incParameter(int p) {
		super.incParameter(p);
		mapNeedsUpdate = true;
	}
	public void decParameter(int p) {
		super.decParameter(p);
		mapNeedsUpdate = true;
	}

	public IplImage process(IplImage src) {
		if(mapNeedsUpdate) {
			createMap();
		}

		if(mapx == null) {
			cvCopy(src,dst);
		} else {
			cvZero(dst);
			cvRemap(src, dst, mapx, mapy,0,BLACK);		
		}
		return dst;
	}



}
