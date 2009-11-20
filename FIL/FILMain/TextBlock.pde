public class TextBlock {
  
  // TEXTBLOCK.pde
  // Controls the visual appearance and collision functionality for text on screen.
  // Extended by Author and Quote classes.
  
  // TEMPORAL VARIABLES
  protected int id;
  protected float x, y;
  protected float w, h;
  protected float redVal             = 0;
  protected float greenVal           = 150;
  protected float blueVal            = 255;
  protected float alphaVal           = 255;
  protected float alphaStartVal      = 255;
  protected float alphaFallOffVal    = 255;
  protected float horizontalFallOff  = 200;
  protected color c;
  
  // TEXT VARIABLES
  protected String textValue, uppercaseTextValue, lowercaseTextValue;
  protected float textScale;
  protected float defaultTextScale;
  protected float maxTextScale;
  protected VTextRenderer textRender;
  protected String fontName;
  protected int fontSize;
  protected float xMargin, yMargin;
  
  // MOVEMENT VARIABLES
  protected float xv = random(-1.0,1.0);  // x velocity
  protected float yv = random(-1.0,1.0);  // y velocity
  protected float xdamping = 0.97;        // horizontal motion damping
  protected float ydamping = 0.97;        // vertical motion damping
  private float gAngle;                   // Angle to gravity center in degrees
  private float gTheta;                   // Angle to gravity center in radians
  private float gxv;                      // Gravity velocity along x axis
  private float gyv;                      // Gravity velocity along y axis
  protected float horizontalSpring;       // multiplier for amount of movement away from each other
  protected float verticalSpring;
  protected int stageWidth, stageHeight;
  protected int counter;
  
  // MODE VARIABLES
  protected boolean fadeIn         = false;
  protected boolean hold           = false;
  protected boolean fadeOut        = false;
  protected boolean scaleUp        = false;
  protected boolean scaleDown      = false;
  protected boolean rollOverEffect = false;
  protected boolean introDelay     = false;
  protected boolean uppercase      = true;
  public boolean remove            = false;
  
  // EFFECT VARIABLES
  protected int rollOverDuration;
  protected int rollOutDuration;
  protected float rollOverScale;
  protected float rollOverAlpha;
  protected int fadeInDuration;
  protected int holdDuration;
  protected int fadeOutDuration;
  
  // INTERACTION VARIABLES
  protected boolean pressed = false;    // triggered true when mouseDown is detected while over this textblock
  
  
  
  
  
  public TextBlock(int id, String textValue, float x, float y, String fontName, int fontSize, float textScale){
    this.id = id;
    this.lowercaseTextValue = textValue;
    this.textValue = this.uppercaseTextValue = textValue.toUpperCase();
    uppercaseTextValue = textValue.toUpperCase();
    this.x = x;
    this.y = y;
    this.fontName = fontName;
    this.fontSize = fontSize;
    this.textScale = textScale;
    this.defaultTextScale = textScale;
    
    textRender = new VTextRenderer(fontName, fontSize);
    textRender.setColor( 1, 1, 1, 1 );
    w = textRender.getWidth(this.textValue) * textScale;
    h = textRender.getHeight(this.textValue) * textScale;
  }
  
  public TextBlock(){
    
  }
  
  
  
  
  
  
  // PHYSICAL ARRANGEMENT FUNCTIONS
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    gAngle        = -radians(findAngle(x,y,xGrav,yGrav));
    gxv           = cos(gAngle) * xgravity;
    gyv           = sin(gAngle) * ygravity;
    xv += gxv;
    yv += gyv;
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5 + b.getWidth()*0.5) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5 + b.getHeight()*0.5) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5)+xMargin + (b.getWidth()*0.5)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5)+yMargin + (b.getHeight()*0.5)+yMargin) - abs(y - b.getY());
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              yv += yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - yoverlap * 0.5 * verticalSpring * textScale);
            } else {
              yv += xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - xoverlap * 0.5 * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              yv -= yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, yoverlap * 0.5 * verticalSpring * textScale);
            } else {
              yv -= xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, xoverlap * 0.5 * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
              xv += yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - yoverlap * 0.5 * horizontalSpring * textScale, 0);
            } else {
              xv += xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - xoverlap * 0.5 * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              xv -= yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(yoverlap * 0.5 * horizontalSpring * textScale, 0);
            } else {
              xv -= xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(xoverlap * 0.5 * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void clearArea(ConcurrentHashMap textBlocks, float multiplier){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){
        if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
          if(b.getX() > x){
              float xoverlap = (x+(w*0.5)) - (b.getX()-(b.getWidth()*0.5));
              b.push((xoverlap / (w*0.5)) * multiplier, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5)) - (x-(w*0.5));
              b.push((xoverlap / (w*0.5)) * multiplier, 0);
            }
        }
      } 
    }
  }
  
  public void clearAbove(ConcurrentHashMap textBlocks, float distance){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){
        if(b.getY() < y && b.getY() >= y - distance){  // if "over" text block...
          if(b.getX()+(b.getWidth()*0.5) > x-(w*0.5) && b.getX()-(b.getWidth()*0.5) < x+(w*0.5)){
            //println(b.getText() +" moved from above "+textValue);
            if(b.getX() > x){
              float xoverlap = (x+(w*0.5)) - (b.getX()-(b.getWidth()*0.5));
              //b.xv += (xoverlap / (w*0.5)) * 5;    // TODO: get multiplier from properties
              b.push((xoverlap / (w*0.5)) * 5, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5)) - (x-(w*0.5));
              //b.xv -= (xoverlap / (w*0.5)) * 5;
              b.push((xoverlap / (w*0.5)) * 5, 0);
            }
          }
        }
      }
    }
  }
  
  public void clearUnderneath(ConcurrentHashMap textBlocks, float distance){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){
        if(b.getY() > y && b.getY() <= y + distance){  // if under text block...
          if(b.getX()+(b.getWidth()*0.5) > x-(w*0.5) && b.getX()-(b.getWidth()*0.5) < x+(w*0.5)){
            //println(b.getText() +" moved from underneath "+textValue);
            if(b.getX() > x){
              float xoverlap = (x+(w*0.5)) - (b.getX()-(b.getWidth()*0.5));
              //b.xv += (xoverlap / (w*0.5)) * 5;    // TODO: get multiplier from properties
              b.push((xoverlap / (w*0.5)) * 5, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5)) - (x-(w*0.5));
              //b.xv -= (xoverlap / (w*0.5)) * 5;
              b.push((xoverlap / (w*0.5)) * 5, 0);
            }
          }
        }
      }
    }
  }
  
  public void fadeOutAndRemove(){
    if(!fadeOut){
      hold = false;
      fadeIn = false;
      introDelay = false;
      fadeOut = true;
      counter = 0;
    }
  }
  
  public void fadeOutAndRemove(int frameDelay){
    if(!fadeOut){
      holdDuration = frameDelay;
      hold = true;
      fadeIn = false;
      introDelay = false;
      counter = 0;
    }
  }
  
  public void removeNow(){
    remove = true;
  }
  
  private float findDistance(float x1, float y1, float x2, float y2){
    float xd = x1 - x2;
    float yd = y1 - y2;
    float td = sqrt(xd * xd + yd * yd);
    return td;
  }
    
  private float findAngle(float x1, float y1, float x2, float y2){
    float xd = x1 - x2;
    float yd = y1 - y2;
  
    float t = atan2(yd,xd);
    float a = (180 + (-(180 * t) / PI));
    return a;
  }
  
  public void move(int stageWidth, int overflow){
    x += xv;
    y += yv;
    xv *= xdamping;
    yv *= ydamping;
    if(x-(w*0.5) > stageWidth + overflow){    // wrap text block to other side of the screen
      x = (0-overflow)+10;
    } else if(x+(w*0.5) < -overflow) {
      x = stageWidth+overflow-10;
    }
  }
  
  
  
  
  
  // RENDERING METHODS
  
  public void drawBoundingBox(int xoffset, int yoffset){
    pushMatrix();
    translate(x-xoffset, y-yoffset, 0);
    stroke(255);
    noFill();
    rect(0-(w*0.5),0-(h*0.5), w, h);  // bounding box around text
    stroke(100);
    rect(0-((w*0.5) + xMargin),0-((h*0.5) + yMargin), w+(xMargin*2), h+(yMargin*2));  // margin bounding box
    popMatrix();
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    pushMatrix();
    // THIS TRANSLATE REQUIRED FLIPPING WHEN SWITCHED TO MPE
    if(yflip){
      translate(x-xoffset, (stageHeight-y) - yoffset, 0);
    } else {
      translate(x-xoffset, y-yoffset, 0);
    }
    rotateX(radians(180));// fix for text getting rendered upside down for some reason
    textRender.setColor( red(c)/255, green(c)/255, blue(c)/255, alpha(c)/255);
    pgl.beginGL();
    if(uppercase){
      textRender.print( uppercaseTextValue, 0-(w*0.5),0-(h*0.5),0,textScale);
    } else {
      textRender.print( lowercaseTextValue, 0-(w*0.5),0-(h*0.5),0,textScale);
    }
    pgl.endGL();
    popMatrix();
  }
  
  
  
  
  
  // PROPERTY RETRIEVAL METHODS
  
  public float getX(){
    return x;
  }
  public float getY(){
    return y;
  }
  
  public float getXVel(){
    return xv;
  }
  public float getYVel(){
    return yv;
  }
  
  public float getWidth(){
    return w;
  }
  public float getHeight(){
    return h; 
  }
  
  // returns potential max dimension given maxTextScale
  public float getMaxWidth(){
    return textRender.getWidth(this.textValue) * maxTextScale;
  }
  public float getMaxHeight(){
    return textRender.getHeight(this.textValue) * maxTextScale;
  }
  
  public float getAlpha(){
    return alphaVal;
  }
  
  public float getScale(){
    return textScale;
  }
  
  public String getText(){
    return textValue;
  }
  
  public VTextRenderer getTextRenderer(){
    return textRender;
  }
  
  public int getID(){
    return id;
  }
  
  
  
  
  
  // INTERACTION METHODS
  
  public boolean isOver(int mx, int my){
    if(mx > x-(w/2) && mx < x+(w/2) && my > y-(h/2) && my < y+(h/2)){
      return true;
    }
    return false;
  }
  
  public void press(){
    pressed = true;
  }
  
  public void release(){
    pressed = false;
  }
  
  public void releasedOutside(){
    pressed = false;
  }
  
  public void rollOver(){
    // TODO: trigger rollOver effect
  }
  
  public void rollOut(){
    
  }
  
  
  
  
  
  // APPEARANCE/POSITION CHANGE METHODS
  
  public void push(float xforce, float yforce){
    xv += xforce;
    yv += yforce;
  }
  
  public void setDamping(float damping){
    this.xdamping = damping;
    this.ydamping = damping;
  }
  public void setHorizontalDamping(float xdamping){
    this.xdamping = xdamping;
  }
  public void setVerticalDamping(float ydamping){
    this.ydamping = ydamping;
  }
  
  public void setScale(float textScale){
    this.textScale = textScale;
    this.defaultTextScale = textScale;
    w = textRender.getWidth(this.textValue) * textScale;
    h = textRender.getHeight(this.textValue) * textScale;
  }
  
  public void setMaxScale(float maxTextScale){
    this.maxTextScale = maxTextScale;
  }
  
  public void setHorizontalSpring(float spring){
    horizontalSpring = spring;
  }
  public void setVerticalSpring(float spring){
    verticalSpring = spring;
  }
  
  public void setX(float x){
    this.x = x;
  }
  public void setY(float y){
    this.y = y;
  }
  
  public void setXMargin(float xMargin){
    this.xMargin = xMargin;
  }
  public void setYMargin(float yMargin){
    this.yMargin = yMargin;
  }
  
  public void setRollOverDuration(int rollOverDuration){
    this.rollOverDuration = rollOverDuration;
  }
  public void setRollOutDuration(int rollOutDuration){
    this.rollOutDuration = rollOutDuration;
  }
  
  public void setRollOverScale(float rollOverScale){
    this.rollOverScale = rollOverScale;
  }
  public void setRollOverAlpha(float rollOverAlpha){
    this.rollOverAlpha = rollOverAlpha;
  }
  
  public void setStageWidth(int stageWidth){
    this.stageWidth = stageWidth;
  }
  public void setStageHeight(int stageHeight){
    this.stageHeight = stageHeight;
  }
  
  
  
  // COLOR CONTROL METHODS FROM SLIDERS
  
  public void setRed(float redVal){  
    this.redVal = redVal;
  }
  public void setGreen(float greenVal){
    this.greenVal = greenVal;
  }
  public void setBlue(float blueVal){
    this.blueVal = blueVal;
  }
  public void setAlphaMax(float alphaStartVal){
    this.alphaStartVal = alphaStartVal;
  }
  public void setAlphaFallOff(float alphaFallOffVal){
    this.alphaFallOffVal = alphaFallOffVal;
  }
  public void setHorizontalFallOff(float horizontalFallOff){
    this.horizontalFallOff = horizontalFallOff;  // distance for alpha fade to the sides
  }
  
}
