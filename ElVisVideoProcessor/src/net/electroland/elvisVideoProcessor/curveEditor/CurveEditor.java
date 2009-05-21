package net.electroland.elvisVideoProcessor.curveEditor;


import java.awt.Color;
import java.awt.FileDialog;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Menu;
import java.awt.MenuBar;
import java.awt.MenuItem;
import java.awt.MenuShortcut;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.CubicCurve2D;
import java.awt.geom.Point2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;


public class CurveEditor extends Frame implements MouseListener, MouseMotionListener, ActionListener {

	Vector<LutChangeListener> lutListeners = new Vector<LutChangeListener>();

	public static final int margin = 70;
	int mouseX;
	int mouseY;
	int lastX;
	int lastY;
	boolean mouseDown = false;

	boolean drawLut = true;
	int lutSize = 255;

	Vector<Curve> curves = new Vector<Curve>();

//	MenuItem toggle;
	MenuItem open;
	MenuItem save;
	MenuItem saveAs;

	int[] lut;
	Polygon lutPoly;

	Rectangle curveBounds;

	
	public CurveEditor() {
		super("Curve Editor");
		setResizable(true);
		MenuBar menubar = new MenuBar();
		Menu menu = new Menu("File");
//		toggle = new MenuItem("ToggleScale");
//		toggle.setShortcut(new MenuShortcut(KeyEvent.VK_T));
//		toggle.addActionListener(this);
		open = new MenuItem("Open...");
		open.setShortcut(new MenuShortcut(KeyEvent.VK_O));
		open.addActionListener(this);
		menu.add(open);

		save = new MenuItem("Save...");
		save.setShortcut(new MenuShortcut(KeyEvent.VK_S));
		save.addActionListener(this);
		menu.add(save);

		saveAs = new MenuItem("Save as LUT...");
		saveAs.setShortcut(new MenuShortcut(KeyEvent.VK_S, true));
		saveAs.addActionListener(this);
		menu.add(saveAs);

//		menu.add(toggle);
		menubar.add(menu);
		this.setMenuBar(menubar);

		setSize(600, 600);
		curveBounds = new Rectangle(margin, margin, getWidth()-margin -margin, getHeight()-margin-margin);
		curves.add(new Curve(new CubicCurve2D.Double(
				curveBounds.x, curveBounds.y+ curveBounds.height, 
				curveBounds.x + curveBounds.width * .25, curveBounds.y + curveBounds.height*.75,
				curveBounds.x + curveBounds.width * .75, curveBounds.y + curveBounds.height*.25,
				curveBounds.x + curveBounds.width, curveBounds.y )));

		this.addMouseMotionListener(this);
		this.addMouseListener(this);

		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent evt) {

				// Hide the frame by setting its visibility as false
				setVisible(false);


			}
		});
	}
	public CurveEditor(String curveFile) throws IOException {
		this();
		open(curveFile);
	}

	public void paint(Graphics g) {
		Graphics2D g2d = (Graphics2D) g;
		if (drawLut) drawLut(g2d);

		g2d.draw(curveBounds);
		g.drawString(""+lutSize, getWidth()-margin -25, getHeight()-margin + 15 );
		boolean endPointSelected = false;
		boolean selectionFree = mouseDown;
		for(Curve curve : curves) {
			if(curve.render(g2d, mouseX, mouseY, lastX,lastY, selectionFree || endPointSelected)) {
				lutChanged = true;
				if(curve.getSelected() == Curve.SELECTED.p2) {
					endPointSelected = true;
					selectionFree = false;
				} else {
					selectionFree = false;
					endPointSelected = false;
				}
			}
		}

	}

	public void addLutChangeListener(LutChangeListener lcl) {
		lutListeners.add(lcl);
	}
	public void removeLutChangeListener(LutChangeListener lcl) {
		lutListeners.remove(lcl);
	}

	public void notifyLutChangeListener(int[] lut) {
		System.out.println("nofiying");
		for(LutChangeListener lcl : lutListeners) {
			lcl.lutChanged();
		}
	}


	public void mouseClicked(MouseEvent e) {
		boolean found = false;
		int i = 0;
		for(i = 0; (i < curves.size()) && (! found); i++) {
			found = curves.get(i).midPoint.distanceSq(e.getX(), e.getY()) < Curve.handelRSqr;
			/*
				found = (e.getX() > curves.get(i).cubcurve.getX1()) &&
				(e.getX() < curves.get(i).cubcurve.getX2());
			 */
		}
		if(found) {
			i--;
			Curve c = curves.remove(i);
			CubicCurve2D cc = c.cubcurve;
			CubicCurve2D left = new CubicCurve2D.Double(0,0,0,0,0,0,0,0);
			CubicCurve2D right = new CubicCurve2D.Double(0,0,0,0,0,0,0,0);
			cc.subdivide(left, right);
			Curve tmp = new Curve(right);
			curves.insertElementAt(tmp, i);
			tmp = new Curve(left);
			curves.insertElementAt(tmp, i);
		}
	} 




	public void drawLut(Graphics2D g) {
		if(lut == null) {
			lut =getLut(lutSize);
			float scaleX = (float) curveBounds.width / (float)lutSize; 
			float scaleY = (float) curveBounds.height /(float)lutSize; 
			lutPoly = new Polygon();
			lutPoly.addPoint(curveBounds.x+curveBounds.width, curveBounds.y + curveBounds.height);
			lutPoly.addPoint(curveBounds.x, curveBounds.y + curveBounds.height);
			for(int i = 0; i < lut.length; i++) {
				int x = (int) (scaleX*i)+curveBounds.x;
				int y = curveBounds.y + curveBounds.height - (int)(scaleY*lut[i]);
				lutPoly.addPoint(x, y);
			}
		}

		g.setColor(Color.LIGHT_GRAY);
		g.fillPolygon(lutPoly);
	}
	
	public int[] getLut(int lutSize) {
		return getLut(lutSize, curves, curveBounds);
	}
	public static int[] getLut(int lutSize, Vector<Curve> curves, Rectangle curveBounds) {
		double xScale = 1.0/curveBounds.getWidth();
		double yScale = 1.0/curveBounds.getHeight();

		int[] result = new int[lutSize];
		for(int i = 0; i < result.length; i++) {
			result[i] = -1;

		}
		for(Curve c : curves) {
			Vector<Point2D> points = c.getPoints(1.0/ (double)(lutSize));
			for(Point2D point : points) {
				int x =(int)(((point.getX()-curveBounds.x) *xScale) * (lutSize));
				int y = (int) (((( (curveBounds.y+curveBounds.height) - point.getY() )*yScale)*(lutSize)));
				//System.out.println(y);
				if((x >= 0) && (x<result.length)) {
					result[x] = (result[x] > y) ? result[x] : y;
				}
			}
		}

		// look for holes and interp
		int holeStart = -1;

		for(int i = 0; i< result.length; i++) {
			if(holeStart >=0) {
				if(result[i]>=0) {
					interp(result, holeStart, i);
					holeStart = -1;
				}
			} else if(result[i] < 0) {
				holeStart = i;
			}

		}
		if(holeStart >= 0) {
			interp(result, holeStart, result.length-1);
		}
		return result;
	}

	public static void interp(int[] ar, int start, int end) {
		if(start == 0) {
			for(int i =0; i<end;i++) {
				ar[i] = ar[end];
			}
		} else if(end >= ar.length) {
			for(int i = start; i <ar.length;i++) {
				ar[i] = ar[start-1];
			}
		} else {
			for(int i = start; i < end;i++) {
				ar[i] = ar[start-1];
			}			
		}

	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mousePressed(MouseEvent e) {
	}

	boolean lutChanged = false;

	public void mouseReleased(MouseEvent e) {
		mouseDown = false;
		if(lutChanged) {
			lut = null;
			repaint();
			notifyLutChangeListener(lut);
		}
	}

	public void mouseDragged(MouseEvent e) {
		if(mouseDown) {
			lastX = mouseX;
			lastY = mouseY;
			mouseX = e.getX();
			mouseY = e.getY();			
		} else {
			mouseX = e.getX();
			mouseY = e.getY();			
			lastX = mouseX;
			lastY = mouseY;
			mouseDown = true;
		}

		repaint();

	}

	public void mouseMoved(MouseEvent e) {
		mouseX = e.getX();
		mouseY = e.getY();
		repaint();

	}

	public static int[] constructLutFromFile(int lutSize, String file) throws IOException {
		Vector<Curve> curves = new Vector<Curve>();
		BufferedReader in = new BufferedReader(new FileReader(file));	
		String line = in.readLine();
		String pnts[] = line.split(",");
		int x = Integer.parseInt(pnts[0]) ;
		int y = Integer.parseInt(pnts[1]) ;
		int w = Integer.parseInt(pnts[2]) ;
		int h = Integer.parseInt(pnts[3]) ;
		Rectangle curveBounds = new Rectangle(x, y, w, h);
		line = in.readLine();
		while(line != null) {
			pnts = line.split(",");
			int i = 0;
			while(i < pnts.length) {
				double px1 = Double.parseDouble(pnts[i++]);
				double py1 = Double.parseDouble(pnts[i++]);
				double cx1 = Double.parseDouble(pnts[i++]);
				double cy1 = Double.parseDouble(pnts[i++]);
				double cx2 = Double.parseDouble(pnts[i++]);
				double cy2 = Double.parseDouble(pnts[i++]);
				double px2 = Double.parseDouble(pnts[i++]);
				double py2 = Double.parseDouble(pnts[i++]);
				curves.add(new Curve(new CubicCurve2D.Double(px1,py1,cx1,cy1,cx2,cy2,px2,py2)));				 
			}
			line = in.readLine();			 
		}
		
		return getLut(lutSize, curves, curveBounds);

	}

	public void open(String file) throws IOException {
		curves.clear();
		BufferedReader in = new BufferedReader(new FileReader(file));	
		String line = in.readLine();
		String pnts[] = line.split(",");
		int x = Integer.parseInt(pnts[0]) ;
		int y = Integer.parseInt(pnts[1]) ;
		int w = Integer.parseInt(pnts[2]) ;
		int h = Integer.parseInt(pnts[3]) ;
		curveBounds.setBounds(x, y, w, h);
		line = in.readLine();
		while(line != null) {
			pnts = line.split(",");
			int i = 0;
			while(i < pnts.length) {
				double px1 = Double.parseDouble(pnts[i++]);
				double py1 = Double.parseDouble(pnts[i++]);
				double cx1 = Double.parseDouble(pnts[i++]);
				double cy1 = Double.parseDouble(pnts[i++]);
				double cx2 = Double.parseDouble(pnts[i++]);
				double cy2 = Double.parseDouble(pnts[i++]);
				double px2 = Double.parseDouble(pnts[i++]);
				double py2 = Double.parseDouble(pnts[i++]);
				curves.add(new Curve(new CubicCurve2D.Double(px1,py1,cx1,cy1,cx2,cy2,px2,py2)));				 
			}
			line = in.readLine();			 
		}
		lut = null;

		repaint();
	}
	public void save(String file) throws IOException {
		FileWriter fw = new FileWriter(file) ;
		StringBuffer sb = new StringBuffer();
		sb.append(curveBounds.x);
		sb.append(",");
		sb.append(curveBounds.y);
		sb.append(",");
		sb.append(curveBounds.width);
		sb.append(",");
		sb.append(curveBounds.height);
		sb.append("\n");
		fw.write(sb.toString());
		for(Curve c : curves) {
			sb = new StringBuffer();
			sb.append(c.cubcurve.getP1().getX());
			sb.append(",");
			sb.append(c.cubcurve.getP1().getY());
			sb.append(",");
			sb.append(c.cubcurve.getCtrlX1());
			sb.append(",");
			sb.append(c.cubcurve.getCtrlY1());
			sb.append(",");
			sb.append(c.cubcurve.getCtrlX2());
			sb.append(",");
			sb.append(c.cubcurve.getCtrlY2());
			sb.append(",");
			sb.append(c.cubcurve.getP2().getX());
			sb.append(",");
			sb.append(c.cubcurve.getP2().getY());
			sb.append("\n");
			fw.write(sb.toString());
		}
		fw.close();
	}

	public void saveAsLut(String file) throws IOException  {
		FileWriter fw = new FileWriter(file) ;
		int[] lut =getLut(256);
		for(int i : lut) {
			fw.write(Integer.toString(i));
			fw.write("\n");
		}
		fw.close();



	}

	public void actionPerformed(ActionEvent e) {
		if(e.getSource() == open) {
			FileDialog fs = new FileDialog(this, "Open Curve", FileDialog.LOAD);
			fs.setVisible(true);
			fs.getFile();
			try {
				open(fs.getDirectory() + File.separator + fs.getFile());
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		} if(e.getSource() == save) {
			FileDialog fs = new FileDialog(this, "Save Curve", FileDialog.SAVE);
			fs.setFile("untitled.elc");
			fs.setVisible(true);
			fs.getFile();
			try {
				save(fs.getDirectory() + File.separator + fs.getFile());
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		} else if(e.getSource() == saveAs) {
			FileDialog fs = new FileDialog(this, "Save Lut 255", FileDialog.SAVE);
			fs.setFile("untitled.lut");
			fs.setVisible(true);
			fs.getFile();
			try {
				saveAsLut(fs.getDirectory() + File.separator + fs.getFile());
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

	}

	public static void main(String[] args) {
		(new CurveEditor()).setVisible(true);
	}
	
	





}