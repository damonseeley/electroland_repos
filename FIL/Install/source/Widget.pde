public abstract class Widget {
  
  // WIDGET.pde
  // Used to capture all types of cursor interaction and act as a super class for all interface items.
  
  private ArrayList listeners;
  protected String name;
  protected int x, y, w, h;
  protected float value;
  protected boolean mouseOver           = false;
  protected boolean mouseDown           = false;
  protected color backgroundColor       = color(50, 50, 50, 200);
  protected color foregroundColor       = color(100, 100, 100, 200);
  protected color activeColor           = color(200, 0, 0, 255);
  protected color activeForegroundColor = color(255, 0, 0, 255);
  protected final int PRESSED = 0;
  protected final int RELEASED = 1;
  protected final int ROLLOVER = 2;
  protected final int ROLLOUT = 3;
  protected final int DRAGGED = 4;
  
  public Widget(String name, int x, int y, float value){
    this.name = name;
    this.x = x;
    this.y = y;
    this.value = value;
    listeners = new ArrayList();
  }
  
  public Widget(String name, int x, int y, int w, int h, float value){
    this.name = name;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.value = value;
    listeners = new ArrayList();
  } 
  
  final public void addListener(WidgetListener wl){
    listeners.add(wl);
  }
	
  final public void removeListener(WidgetListener wl){
    listeners.remove(wl);
  }
	
  final public void newEvent(WidgetEvent we){
    Iterator i = listeners.iterator();
    while (i.hasNext()){
      ((WidgetListener)i.next()).widgetEvent(we);
    }
  }
  
  
  /* MOUSE EVENT FUNCTIONS */
	
  public abstract void draw();
  public abstract void pressed();
  public abstract void released();
  public abstract void dragged();
  public abstract void rollOver();
  public abstract void rollOut();
  public abstract void cursorMovement();
  
  final public void mouseMoved(int mouseX, int mouseY){
    if(mouseInside(mouseX, mouseY)){	// if within constraints, activate rollOver
      if(!mouseOver){
        mouseOver = true;
        rollOver();
      }
    } else {				// if outside constraints, activate rollOut
      if(mouseOver){
        mouseOver = false;
        rollOut();
      }
    }
    cursorMovement();			// verbose movement repeater (needed for embedded items)
  }
	
  final public void mouseDragged(int mouseX, int mouseY){
    if(mouseDown){
      dragged();
    }
  }
	
  final public void mousePressed(int mouseX, int mouseY){
    if(mouseInside(mouseX, mouseY)){
      mouseDown = true;
      pressed();
    }
  }

  final public void mouseReleased(int mouseX, int mouseY){
    if(mouseDown){
      released();
      mouseDown = false;
    }
  }
	
  final public boolean mouseInside(int mouseX, int mouseY){
    if((mouseX >= x && mouseX <= x+w) && (mouseY >= y && mouseY <= y+h)){
      return true;
    } else {
      return false;
    }
  }
	
  final public String getName(){
    return name;
  }
  
  final public int getX(){
    return x;
  }
  
  final public int getY(){
    return y;
  }
  
  final public int getWidth(){
    return w;
  }
  
  final public int getHeight(){
    return h;
  }
  
  final public void setX(int x){
    this.x = x;
  }
  
  final public void setY(int y){
    this.y = y;
  }
  
  
  
  /* COLOR SETTER FUNCTIONS */
	
  final public void setBackgroundColor(color c){
    backgroundColor = c;
  }
	
  final public void setForegroundColor(color c){
    foregroundColor = c;
  }
	
  final public void setActiveColor(color c){
    activeColor = c;
  }
	
  final public void setActiveForegroundColor(color c){
    activeForegroundColor = c;
  }
	
  final public float getValue(){
    return value;
  }
  
}
