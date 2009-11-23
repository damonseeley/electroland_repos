public class Balloon extends Widget implements WidgetListener{
  
  private PImage img, imgDown;
  private PImage quoteImg, quoteDown, quoteImgEsp, quoteDownEsp, quoteGrey, bioImg, bioDown, bioImgEsp, bioDownEsp, bioGrey;
  private Button quote;
  private Button bio;
  public Author author;
  public boolean quoteMode = true;
  public boolean bioMode = false;
  private boolean fadeout = false;
  private int counter = 0;
  private int fadeOutDuration;
  private float horizontalOffset, verticalOffset;
  private float interfaceScale;
  private boolean flipped = false;
  private String userLanguage;
  
  public Balloon(String name, Author author, float value, PImage img, PImage imgDown, PImage quoteImg, PImage quoteDown, PImage quoteImgEsp, PImage quoteDownEsp, PImage quoteGrey, PImage bioImg, PImage bioDown, PImage bioImgEsp, PImage bioDownEsp, PImage bioGrey, float interfaceScale, float horizontalOffset, float verticalOffset, String userLanguage){
    super(name, int(author.getX()*(1/interfaceScale)), int((author.getY()+verticalOffset)*(1/interfaceScale)) - img.height, value);
    this.author = author;
    this.w = img.width;
    this.h = img.height;
    this.img = img;
    this.imgDown = imgDown;
    this.quoteImg = quoteImg;
    this.quoteDown = quoteDown;
    this.quoteImgEsp = quoteImgEsp;
    this.quoteDownEsp = quoteDownEsp;
    this.quoteGrey = quoteGrey;
    this.bioImg = bioImg;
    this.bioDown = bioDown;
    this.bioImgEsp = bioImgEsp;
    this.bioDownEsp = bioDownEsp;
    this.bioGrey = bioGrey;
    this.interfaceScale = interfaceScale;
    this.horizontalOffset = horizontalOffset;
    this.verticalOffset = verticalOffset;
    this.userLanguage = userLanguage;
    
    x = int(((author.getX()-(author.getWidth()/2)) / interfaceScale) - horizontalOffset) + 10 - img.width;
    y = int((author.getY() / interfaceScale) - verticalOffset) + 10 - img.height;
    int xpos = 0;
    if(x < 0){
      flipped = true;
      xpos = img.width - quoteImg.width;
    }
    
    if(userLanguage == "English"){
      if(author.hasQuote(userLanguage)){
        quote = new Button("Quote", xpos, 0, 0, quoteImg, quoteDown);
        quote.silentOn();
      } else {
        quote = new Button("Quote", xpos, 0, 0, quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio = new Button("Biography", xpos, 48, 0, bioImg, bioDown);
      } else {
        bio = new Button("Biography", xpos, 48, 0, bioGrey, bioGrey);
      }
    } else {
      if(author.hasQuote(userLanguage)){
        quote = new Button("Quote", xpos, 0, 0, quoteImgEsp, quoteDownEsp);
        quote.silentOn();
      } else {
        quote = new Button("Quote", xpos, 0, 0, quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio = new Button("Biography", xpos, 48, 0, bioImgEsp, bioDownEsp);
      } else {
        bio = new Button("Biography", xpos, 48, 0, bioGrey, bioGrey);
      }
    }
    
    //quote = new Button("Quote", xpos, 0, 0, quoteImg, quoteDown);
    quote.addListener(this);
    //bio = new Button("Biography", xpos, 48, 0, bioImg, bioDown);
    bio.addListener(this);
  }
  
  public void draw(){
    pushMatrix();
    
    if(flipped){
      x = ((author.getX()+(author.getWidth()/2)) / interfaceScale) - horizontalOffset;
    } else {
      x = ((author.getX()-(author.getWidth()/2)) / interfaceScale) - horizontalOffset + 10 - img.width;
    }
    y = (author.getY() / interfaceScale) - verticalOffset + 10 - img.height;
    
    translate(x, y);
    if(fadeout){
      float progress = sin((counter / (float)fadeOutDuration) * (PI/2));
      tint(255, 255 - (progress*255));
      counter++;
    }
    if(flipped){
      pushMatrix();
      scale(-1,1,1);
      if(quoteMode){
        image(img,0-img.width,0);
      } else {
        image(imgDown,0-img.width,0);
      }
      popMatrix();
    } else {
      if(quoteMode){
        image(img,0,0);
      } else {
        image(imgDown,0,0);
      }
    }
    quote.draw();
    bio.draw();
    if(fadeout){
      tint(255, 255);
    }
    popMatrix();
  }
  
  public void dragged(){
    // probably need to inform drop down items from here
  }
  
  public void fadeOut(int duration){
    fadeout = true;
    fadeOutDuration = duration;
  }
  
  public void pressed(){
    if(!fadeout){
      quote.mousePressed(mouseX-x, mouseY-y);
      bio.mousePressed(mouseX-x, mouseY-y);
    }
  }
  
  public void released(){
  }
  
  public void rollOver(){
    //quote.rollOver();
    //bio.rollOver();
  }
  
  public void rollOut(){
    //quote.rollOut();
    //bio.rollOut();
  }
  
  public void cursorMovement(){
    //quote.cursorMovement();
    //bio.cursorMovement();
  }
  
  public void setAuthor(Author author){
    this.author = author;
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    if(we.widget == quote){
      bio.silentOff();
      quoteMode = true;
      bioMode = false;
      // TODO: trigger quote reveal here
    } else {
      quote.silentOff();
      bioMode = true;
      quoteMode = false;
      // TODO: trigger biography reveal here
    }
    super.newEvent(new WidgetEvent(this, PRESSED, true));
  }
  
  public void setInterfaceScale(float interfaceScale){
    this.interfaceScale = interfaceScale;
  }
  
  public void setHorizontalOffset(float horizontalOffset){
    this.horizontalOffset = horizontalOffset;
  }
  
  public void setVerticalOffset(float verticalOffset){
    this.verticalOffset = verticalOffset;
  }
  
  public void setUserLanguage(String userLanguage){
    this.userLanguage = userLanguage;
    if(userLanguage == "English"){
      if(author.hasQuote(userLanguage)){
        quote.switchImages(quoteImg, quoteDown);
      } else {
        quote.switchImages(quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio.switchImages(bioImg, bioDown);
      } else {
        bio.switchImages(bioGrey, bioGrey);
      }
    } else {
      if(author.hasQuote(userLanguage)){
        quote.switchImages(quoteImgEsp, quoteDownEsp);
      } else {
        quote.switchImages(quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio.switchImages(bioImgEsp, bioDownEsp);
      } else {
        bio.switchImages(bioGrey, bioGrey);
      }
    }
  }
  
}
