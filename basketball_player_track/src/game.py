'''adapt from 
https://github.com/linouk23/NBA-Player-Movements
https://github.com/sealneaward/movement-quadrants
https://github.com/sealneaward/nba-movement-data
http://savvastjortjoglou.com/nba-play-by-play-movements.html
'''
import pandas as pd

class Player:
    def __init__(self, player_dict):
        '''attributes: playerid, firstname, lastname, jersey, position
        '''
        for key in player_dict:
            setattr(self, key, player_dict[key])
            
    def __repr__(self):
        return f'{self.firstname} {self.lastname}, #{self.jersey}, {self.position}'
        
class Team:
    def __init__(self, team_dict):
        '''attributes: teamid, name, abbreviation, players
        '''
        for key in team_dict:
            setattr(self, key, team_dict[key])
        self.players = [Player(player) for player in self.players]
        
        self.players_mapping = {}
        for player in self.players:
            self.players_mapping[f'{player.firstname} {player.lastname}'] = player.playerid
        
    def __repr__(self):
        descr = f'{self.name}\n\tPlayers:\n'
        for player in self.players:
            descr += f"\t{repr(player)}\n"
        return descr
        
class Game:
    def __init__(self, game_json_file):
        self.game_json_file = game_json_file
        data_frame = pd.read_json(game_json_file)
        self.events = data_frame['events']
        self.num_events = len(self.events)
        
        self.visitor = Team(self.events[0]['visitor'])
        self.home = Team(self.events[0]['home'])
        
        # name to id
        self.players_mapping = {**self.visitor.players_mapping, **self.home.players_mapping}
    
    def __repr__(self):
        descr = 'Visitor: ' + repr(self.visitor)
        descr += 'Home: ' + repr(self.home)
        return descr
    
    def get_player_list(self):
        names = list(self.visitor.players_mapping.keys()) + list(self.home.players_mapping.keys())
        return names
    
    def get_player_movement(self, player_name, event_id):
        playerid = self.players_mapping[player_name]
        event = self.events[event_id]
        moments = event['moments']
        x, y, t = [], [], []
        for moment in moments:
            for player in moment[5]:
                if player[1] == playerid:
                    x.append(player[2])
                    y.append(player[3])
                    t.append(moment[2])
        return x, y, t
        