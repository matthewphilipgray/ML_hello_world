import pygame
from Network import Network
import random 

class App:
    def __init__(self, fps = 30):
        self._running = True
        self.playing = True
        self._display_surf = None
        self.size = self.width, self.height = 800, 480
        self.fps = fps
        pygame.init()
        pygame.display.set_caption("Colour Picker")
        
        function = ["sigmoid", "softmax"]
        self.brain = Network(3,[3],2, function)
        
        self.colour = [random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)]
        inp = [col / 255 for col in self.colour]
        self.brain.feed_forward(inp)
        out = self.brain.output
        self.get_guess(out)
            
        self.training = False
        
 
    def on_init(self):
                
        self.clock = pygame.time.Clock()        
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 60, bold=True)
        self._running = True
        self.playing = True
        
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill(self.colour)
        pygame.draw.line(self.background, (0, 0, 0), (self.width // 2, 0), (self.width // 2, self.height))
        
    def get_guess(self, out):
        
        if out[0] > out[1]:
            self.guess = "black"
        else:
            self.guess = "white"
    
    def run_brain(self, ideal):
    
        self.brain.backpropogate(ideal)
        self.colour = [random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)]
        self.background.fill(self.colour)
        pygame.draw.line(self.background, (0, 0, 0), (self.width // 2, 0), (self.width // 2, self.height))
        inp = [col / 255 for col in self.colour]
        self.brain.feed_forward(inp)
        out = self.brain.output
        self.get_guess(out)
        
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.run_brain([1, 0])
            if event.key == pygame.K_RIGHT:
                self.run_brain([0, 1])
            if event.key == pygame.K_ESCAPE:
                self._running = False
            if event.key == pygame.K_RETURN:
                self.training = not self.training
            if event.key == pygame.K_SPACE:
                print(self.brain.output)
            
    def on_loop(self):
        
        if self.training:
            count = 0
            print("training")
            while count < 100:
                if sum(self.colour) > 380:
                    self.run_brain([1, 0])
                else:
                    self.run_brain([0, 1])
                    
                count = count + 1
        
    
    def on_render(self):
        
        self.screen.blit(self.background, (0, 0))
        black = "black"
        white = "white"
        text = self.font.render(black, True, (0,0,0))
        text_rect = text.get_rect(center=(self.width/4, self.height//2))
        self.screen.blit(text, text_rect)       
        text = self.font.render(white, True, (255,255,255))
        text_rect = text.get_rect(center=(3*self.width/4, self.height//2))
        self.screen.blit(text, text_rect)
        
        
        
        
        point = pygame.Surface((50, 50))
        y = self.height // 4
        x = self.height // 4
        
        if self.guess == "white":
            point.fill((255,255,255))
            x = 3 * self.height // 4
            
        point.get_rect(center=(x,y))
        
        self.screen.blit(point, (x,y))
        
        pygame.display.flip()
        
    
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        
        
        while(self._running):
            
            
            ms = self.clock.tick(self.fps)
            self.playtime += ms *0.001
            
            
            for event in pygame.event.get():
                self.on_event(event)
            
            if self.playing:
                self.on_loop()
                self.on_render()
            else:
                self.playtime = 0.0
                
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App(30)
    theApp.on_execute()



















































