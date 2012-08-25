package net.electroland.elvis.util;

import java.awt.geom.Point2D;
import java.util.Vector;

import com.googlecode.javacv.cpp.opencv_core.CvPoint;


// this code is ugly because its adapted from some c code
public class BilinearInterp {

	public static boolean equals( double a, double b, double tolerance )
	{
		return (( a == b ) ||
				( ( a <= ( b + tolerance ) ) &&
						( a >= ( b - tolerance ))));
	}

	public static double cross2( double x0, double y0, double x1, double y1 )
	{
		return x0*y1 - y0*x1;
	}
	public static boolean in_range( double val, double range_min, double range_max, double tol )
	{
		return ((val+tol) >= range_min) && ((val-tol) <= range_max);
	}


	/* Returns number of solutions found.  If there is one valid solution, it will be put in s and t */
	public static Vector<Point2D> inverseBilerp(CvPoint a, CvPoint b, CvPoint c, CvPoint d, CvPoint p) {
		return inverseBilerp(d.x(), d.y(), c.x(), c.y(), a.x(), a.y(), b.x(), b.y(), p.x(), p.y());
	}
	public static Vector<Point2D> inverseBilerp( double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double x, double y)
	{
		boolean t_valid, t2_valid;

		double a  = cross2( x0-x, y0-y, x0-x2, y0-y2 );
		double b1 = cross2( x0-x, y0-y, x1-x3, y1-y3 );
		double b2 = cross2( x1-x, y1-y, x0-x2, y0-y2 );
		double c  = cross2( x1-x, y1-y, x1-x3, y1-y3 );
		double b  = 0.5 * (b1 + b2);

		double s = 0; 
		double s2= 0;
		double t= 0;
		double t2= 0;

		double am2bpc = a-2*b+c;
		/* this is how many valid s values we have */
		int num_valid_s = 0;

		if ( equals( am2bpc, 0, 1e-10 ) )
		{
			if ( equals( a-c, 0, 1e-10 ) )
			{
				/* Looks like the input is a line */
				/* You could set s=0.5 and solve for t if you wanted to */
				return null;
			}
			s = a / (a-c);
			if ( in_range( s, 0, 1, 1e-10 ) )
				num_valid_s = 1;
		}
		else
		{
			double sqrtbsqmac = Math.sqrt( b*b - a*c );
			s  = ((a-b) - sqrtbsqmac) / am2bpc;
			s2 = ((a-b) + sqrtbsqmac) / am2bpc;
			num_valid_s = 0;
			if ( in_range( s, 0, 1, 1e-10 ) )
			{
				num_valid_s++;
				if ( in_range( s2, 0, 1, 1e-10 ) )
					num_valid_s++;
			}
			else
			{
				if ( in_range( s2, 0, 1, 1e-10 ) )
				{
					num_valid_s++;
					s = s2;
				}
			}
		}

		if ( num_valid_s == 0 )
			return null;

		t_valid = false;
		if ( num_valid_s >= 1 )
		{
			double tdenom_x = (1-s)*(x0-x2) + s*(x1-x3);
			double tdenom_y = (1-s)*(y0-y2) + s*(y1-y3);
			t_valid = true;
			if ( equals( tdenom_x, 0, 1e-10 ) && equals( tdenom_y, 0, 1e-10 ) )
			{
				t_valid = false;
			}
			else
			{
				/* Choose the more robust denominator */
				if ( Math.abs( tdenom_x ) > Math.abs( tdenom_y ) )
				{
					t = ( (1-s)*(x0-x) + s*(x1-x) ) / ( tdenom_x );
				}
				else
				{
					t = ( (1-s)*(y0-y) + s*(y1-y) ) / ( tdenom_y );
				}
				if ( !in_range( t, 0, 1, 1e-10 ) )
					t_valid = false;
			}
		}

		/* Same thing for s2 and t2 */
		t2_valid = false;
		if ( num_valid_s == 2 )
		{
			double tdenom_x = (1-s2)*(x0-x2) + s2*(x1-x3);
			double tdenom_y = (1-s2)*(y0-y2) + s2*(y1-y3);
			t2_valid = true;
			if ( equals( tdenom_x, 0, 1e-10 ) && equals( tdenom_y, 0, 1e-10 ) )
			{
				t2_valid = false;
			}
			else
			{
				/* Choose the more robust denominator */
				if ( Math.abs( tdenom_x ) > Math.abs( tdenom_y ) )
				{
					t2 = ( (1-s2)*(x0-x) + s2*(x1-x) ) / ( tdenom_x );
				}
				else
				{
					t2 = ( (1-s2)*(y0-y) + s2*(y1-y) ) / ( tdenom_y );
				}
				if ( !in_range( t2, 0, 1, 1e-10 ) )
					t2_valid = false;
			}
		}

		/* Final cleanup */
		if ( t2_valid && !t_valid )
		{
			s = s2;
			t = t2;
			t_valid = t2_valid;
			t2_valid = false;
		}

		Vector<Point2D> results = new Vector<Point2D>();
		/* Output */
		if ( t_valid )
		{
			results.add(new Point2D.Double(s,t));
		}

		if ( t2_valid )
		{
			results.add(new Point2D.Double(s2,t2));
		}

		return results;	
	}

	
	public static Point2D bilerp(CvPoint a, CvPoint b, CvPoint c, CvPoint d, Point2D st) {
		return bilerp(d.x(), d.y(), c.x(), c.y(), a.x(), a.y(), b.x(), b.y(), st.getX(), st.getY());
	}


	public static Point2D bilerp( double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3, double s, double t )
	{
		return new Point2D.Double( t*(s*x3+(1-s)*x2) + (1-t)*(s*x1+(1-s)*x0),
				t*(s*y3+(1-s)*y2) + (1-t)*(s*y1+(1-s)*y0));
	}
	

	
}
