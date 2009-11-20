public class Button extends Widget{
  
  private boolean on = false;
  private PImage img;
  private PImage imgDown;
  private boolean imgMode = false;
  private VTextRenderer textRender;
  private float textScale;
  private float textWidth, textHeight;
  
  public Button(String name, int x, int y, int w, int h, float value){
    super(name, x, y, w, h, value);
    String fontName = "HelveticaNeue BoldCond";
    int fontSize    = 16;
    textRender = new VTextRenderer(fontName, fontSize);
    textRender.setColor( 1, 1, 1, 1 );
    textWidth = textRender.getWidth(name);
    textHeight = textRender.getHeight(name);
  }
  
  public Button(String name, int x, int y, int w, int h, float value, PImage img){
    super(name, x, y, w, h, value);
    this.img = img;
    imgMode = true;
  }
  
  public Button(String name, int x, int y, float value, PImage img, PImage imgDown){
    super(name, x, y, value);
    this.img = img;
    this.imgDown = imgDown;
    imgMode = true;
    w = img.width;
    h = img.height;
  }
  
  public void draw(){
    pushMatrix();
    if(imgMode){
      translate(x,y);
      if(on){
        image(imgDown,0,0);
      } else {
        image(img,0,0);
      }
    } else {
      noStroke();
      if(mouseOver && on){
        fill(activeForegroundColor);
      } else if(mouseOver && !on){
        fill(foregroundColor);
      } else if(on){
        fill(activeColor);
      } else {
        fill(backgroundColor);
      }
      rect(0,y,w,h);
      // this should be better
      textRender.print(name, int(x+(w/2)) - int(textWidth/2), (height - 50 - y) - int(h/2 + textHeight/2));
      
    }
    popMatrix();
  }
  
  public void silentToggle(){
    on = !on;
  }
  
  public void silentOff(){
    on = false;
  }
  
  public void silentOn(){
    on = true;
  }
  
  public void pressed() {
    if(!on){
      //on = !on;
      on = true;
      WidgetEvent we = new WidgetEvent(this, PRESSED, on);
      super.newEvent(we);
    }
  }

  public void released() {
  }

  public void rollOut() {
    println("rolled out of "+name);
  }

  public void rollOver() {
    println("rolled over "+name);
  }

  public void cursorMovement() {
  }

  public void dragged() {
    // TODO Auto-generated method stub	
  }	
  
  public void switchImages(PImage img, PImage imgDown){
    this.img = img;
    this.imgDown = imgDown;
  }
  
}
