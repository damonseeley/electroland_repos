/**
 * 
 */
package net.electroland.coopLights.core;

import java.awt.Color;
import java.awt.Graphics;


public class Region {
		int id;
		float top;
		float left;
		float bot;
		float right;
		int w;
		int h;
		Color c;
		
		//NOTE DAMON CHANGED THIS TO REFLECT SCREEN COORD TYPES
		public Region(int id, float left, float top,float right, float bot) {
			this.id = id;
			this.top = top;
			this.left = left;
			this.bot = bot;
			this.right = right;
			w = (int) (right-left);
			h = (int) (bot-top);
			//c = new Color((float)Math.random(), (float)Math.random(), (float)Math.random());
			int blueSub = (int)(Math.random()*64);
			c = new Color(255-blueSub, 255-blueSub, 255);
	}
		
		public void setColor(Color c) { 
			this.c = c;
		}

		public void render(Graphics g) {
			float xScale = InstallSimMain.xScale;
//			float yScale = xScale;  UNUSED EGM
			float xOffset = InstallSimMain.xOffset;
			float yOffset = InstallSimMain.yOffset;
			
			g.setColor(c);
			g.fillRect((int)(left+xOffset),(int)(top+yOffset),w,h);
			g.setColor(new Color(128, 128, 128));
			g.drawString("r: " + this.id, (int)(left+xOffset)+w/2,(int)(top+yOffset)+h/2+4);
			
		}
		
		public String toString() { return "(" + top + ", " + left + ", " + bot + ", " + right + ")"; }
		
		public boolean contains(float x, float y) {
			if(y < top) return false;
			if(y >= bot) return false;
			if(x < left) return false;
			if(x >= right) return false;
			return true;
		}
	}