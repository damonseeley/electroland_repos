package net.electroland.fish.ui;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCanvas;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.swing.JFrame;

import net.electroland.fish.boids.BigFish1;
import net.electroland.fish.boids.InvisibleSoundFish;
import net.electroland.fish.core.Pond;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;

import com.sun.opengl.util.FPSAnimator;

public class MainFrame extends JFrame
implements GLEventListener, KeyListener, MouseListener
{


	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private GLCapabilities caps;
	private GLCanvas canvas;
	private FPSAnimator animator;


	private int screenWidth;
	private int screenHeight;

	private Bounds worldBounds;


	Pond pond ;

	public MainFrame(Pond pond)
	{
		super("Electoland Mira Fish");

		
		this.pond = pond;
		worldBounds = pond.bounds;
		screenWidth = FishProps.THE_FISH_PROPS.getProperty("screenWidth", 800);
		screenHeight = FishProps.THE_FISH_PROPS.getProperty("screenHeight", 600);


		caps = new GLCapabilities();
		caps.setDoubleBuffered(true);// request double buffer display mode
		canvas = new GLCanvas(caps);
		canvas.addGLEventListener(this);
		canvas.addMouseListener(this);// register mouse callback functions
		canvas.addKeyListener(this);
		animator = new FPSAnimator(canvas, 40);

		getContentPane().add(canvas);
		

	}



	public void startRendering()
	{
		setSize(screenWidth, screenHeight);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);
		animator.start();
	}

	
	public void init(GLAutoDrawable drawable)
	{
		GL gl = drawable.getGL();
		gl.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		gl.glShadeModel(GL.GL_FLAT);
		gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
		gl.glEnable(GL.GL_BLEND);
		gl.glEnable(GL.GL_POINT_SMOOTH);
		gl.glEnable(GL.GL_LINE_SMOOTH);
		gl.glEnable( GL.GL_POLYGON_SMOOTH ); 
	}

	public void display(GLAutoDrawable drawable)
	{
		
		GL gl = drawable.getGL();
		gl.glClear(GL.GL_COLOR_BUFFER_BIT);
		gl.glPushMatrix();
//		gl.glRotatef(spin, 0.0f, 0.0f, 1.0f);
		gl.glColor3f(1.0f, 1.0f, 1.0f);

//		gl.glRectf(700f, 500f, 900f, 700f);
//		gl.glRectf(0f, 0f, 100f, 100f);
//		gl.glRectf(1500, 1100, 1600f, 1200f);

		pond.draw(drawable, gl);

		gl.glPopMatrix();

		gl.glFlush();
	}

	public void reshape(GLAutoDrawable drawable, int x, int y, int w, int h)
	{
		GL gl = drawable.getGL();
		gl.glViewport(0, 0, w, h);
		gl.glMatrixMode(GL.GL_PROJECTION);
		gl.glLoadIdentity();

		screenWidth = w;
		screenHeight = h;

		float screenRatio = (float)screenWidth /(float)screenHeight;
		float worldRatio =  worldBounds.getWidth() / worldBounds.getHeight();

		if(screenRatio > worldRatio) { // wide screen have to limit world by height(add more to world width)
			float newWidth = worldBounds.getHeight() * screenRatio;
			newWidth *= .5;
			float worldCenter = (worldBounds.getRight() + worldBounds.getLeft()) * .5f;
			gl.glOrtho(worldCenter - newWidth, worldCenter + newWidth, worldBounds.getBottom(), worldBounds.getTop(), -1.0, 1.0);


		} else { // add more to world height
			float newHeight =  worldBounds.getWidth() / screenRatio;
			newHeight *= .5f;
			float worldCenter = (worldBounds.getBottom() + worldBounds.getTop()) * .5f;

			gl.glOrtho(worldBounds.getLeft(), worldBounds.getRight(), worldCenter+newHeight, worldCenter - newHeight, -1.0, 1.0);

		}

//		gl.glOrtho(worldLeft, worldRight, worldBottom, worldTop, -1.0, 1.0);

		gl.glMatrixMode(GL.GL_MODELVIEW);
		gl.glLoadIdentity();
	}

	public void displayChanged(GLAutoDrawable drawable, boolean modeChanged,
			boolean deviceChanged)
	{
	}


	public void keyTyped(KeyEvent key)
	{
	}

	public void keyPressed(KeyEvent key)
	{
		switch (key.getKeyCode()) {
		case KeyEvent.VK_ESCAPE:
			new Thread()
			{
				public void run()
				{
					quit();
				}
			}.start();
			System.exit(0);
		case KeyEvent.VK_F:	
			System.out.println("fps = " + pond.xmlFrameRateCalc.fps);
			System.out.println("el = " + pond.xmlFrameRateCalc.elapsedTime);
			System.out.println("max er = " + pond.xmlFrameRateCalc.maxVar);
			System.out.println("min er = " + pond.xmlFrameRateCalc.minVar);
			pond.xmlFrameRateCalc.maxVar = 0;
			pond.xmlFrameRateCalc.minVar =0;
			break;
		case KeyEvent.VK_S:
				System.out.println("saving propety file: " + FishProps.THE_FISH_PROPS.fileName);
					FishProps.THE_FISH_PROPS.store();
			break;
		case KeyEvent.VK_W:
			System.out.println("playing wave");
			pond.toggleWave();			
			break;
		case KeyEvent.VK_A:
			System.out.println("playing ambient sound");
			InvisibleSoundFish.playAmbientSound();
			break;
		case KeyEvent.VK_Q:
			System.out.println("muting ambient sound");
			InvisibleSoundFish.muteAmbient();
			break;
		case KeyEvent.VK_1:
		case KeyEvent.VK_2:
		case KeyEvent.VK_3:
		case KeyEvent.VK_4:
		case KeyEvent.VK_5:
		case KeyEvent.VK_6:
		case KeyEvent.VK_7:
		case KeyEvent.VK_8:
		case KeyEvent.VK_9:
			int keyCode = key.getKeyCode() - '1';
			BigFish1.THEBIGFISH.get(keyCode).touched();
			break;
		default:
			break;
		}
	}

	public void quit() {
		animator.stop();	  
		System.exit(0);
	}

	public void keyReleased(KeyEvent key)
	{
	}

	public void mouseClicked(MouseEvent key)
	{
	}

	public void mousePressed(MouseEvent mouse)
	{
		switch (mouse.getButton()) {
		case MouseEvent.BUTTON1:
			break;
		case MouseEvent.BUTTON2:
		case MouseEvent.BUTTON3:
			break;
		}
	}

	public void mouseReleased(MouseEvent mouse)
	{
	}

	public void mouseEntered(MouseEvent mouse)
	{
	}

	public void mouseExited(MouseEvent mouse)
	{
	}

}