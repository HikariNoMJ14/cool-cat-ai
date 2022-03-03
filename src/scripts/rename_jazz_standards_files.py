import os
from glob import glob
import re
import shutil
from string_similarity import similar


def phase_1():
    folder = '../../data/Raw Data/JazzStandards'

    files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]
    files += [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.MID'))]

    print(len(files))

    for file in files:
        f = os.path.basename(file)

        f = re.sub(r"([A-Z][a-z]+)", r"\1 ", f)
        f = f.replace('_', ' ')
        f = f.replace('%20', ' ')
        f = f.replace('  ', ' ')

        upper = False
        g = ''
        for char in f:
            if char == ' ':
                upper = True
            elif upper:
                char = char.upper()
                upper = False
            g += char

        g = g.rstrip()
        g = g.replace(' .mid', '.mid').replace(' .mid', '.mid')

        out_file = os.path.join(os.path.dirname(file).replace('JazzStandards', 'JazzStandards2'), g)

        print(out_file)

        shutil.copyfile(file, out_file)


def phase_2():
    folder = '../../data/Raw Data/JazzStandards2'

    files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mid'))]
    files += [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.MID'))]

    print(len(files))

    renamed = [
        "4 On 6.mid",
        "A Foggy Day In London Town.mid",
        "A Night In Tunisia (1).mid",
        "Afro Blues.mid",
        "Afternoon In Paris.mid",
        "Ain't Misbehaving.mid",
        "Alice's Restaurant.mid",
        "All Blues.mid",
        "All My Tomorrows.mid",
        "All Of Me.mid",
        "All The Things You Are.mid",
        "Alone Together.mid",
        "Along Came Betty.mid",
        "Always With You.mid",
        "Ana Maria.mid",
        "Angel Eyes.mid",
        "A Night In Tunisia (2).mid",
        "Anthropology.mid",
        "Anything Goes.mid",
        "April In Paris.mid",
        "Au Privave.mid",
        "Autumn Leaves.mid",
        "Bags Groove.mid",
        "Bb Blues.mid",
        "Beautiful Love.mid",
        "Beauty And The Beast.mid",
        "Bebop.mid",
        "Bernies Tune.mid",
        "Bessie's Blues.mid",
        "Better Days Ahead.mid",
        "Big Bill Broonzy.mid",
        "Big Time.mid",
        "Billies Bounce.mid",
        "Birdland.mid",
        "Birth Of The Blues.mid",
        "Black Nacissus.mid",
        "Black Orpheus.mid",
        "Blue In Green (1).mid",
        "Blue In Green (2).mid",
        "Blue Monk.mid",
        "Blue Rondo.mid",
        "Blue Train.mid",
        "Blues For Alice.mid",
        "Blues Jam.mid",
        "Bluesette.mid",
        "Body And Soul.mid",
        "Boplicity.mid",
        "But Beautiful.mid",
        "Bye Bye Blackbird.mid",
        "Can't Help Falling In Love.mid",
        "Cantaloupe Islands.mid",
        "Captain Marvel.mid",
        "Chameleon.mid",
        "Chatanoogoo.mid",
        "Chega De Saudade (2).mid",
        "Chega De Saudade (1).mid",
        "Cherokee.mid",
        "Chicken.mid",
        "Chip Blue.mid",
        "Come Rain Or Come Shine.mid",
        "Come Sunday.mid",
        "Con Alma.mid",
        "Count Down.mid",
        "Criss Cross.mid",
        "Darn That Dream.mid",
        "Days Of Wine And Roses (1).mid",
        "Days Of Wine And Roses (2).mid",
        "Desafinado.mid",
        "Dig.mid",
        "Dolphin Dance.mid",
        "Donna Lee (2).mid",
        "Donna Lee (1).mid",
        "Doxy.mid",
        "Esp.mid",
        "East Of The Sun.mid",
        "Eighty One.mid",
        "El Gaucho.mid",
        "End Of The Night.mid",
        "Epistrophy.mid",
        "Equinox.mid",
        "FMinor Blues.mid",
        "Fake It.mid",
        "Fast Live.mid",
        "Footprints.mid",
        "Four Brothers.mid",
        "Four.mid",
        "Freedom Jazz Dance.mid",
        "Giant Steps.mid",
        "Good Bait.mid",
        "Goodbye Porkpie Hat.mid",
        "Green Dolphin Street Easy.mid",
        "Hakensak.mid",
        "Half Nelson.mid",
        "Have You Met Miss Jones.mid",
        "Here's That Rainy Day.mid",
        "Hi-Fly.mid",
        "Hoochie Coochie Man.mid",
        "House Of Jade.mid",
        "How High The Moon.mid",
        "How Insensitive.mid",
        "I Concentrate On You.mid",
        "I Could Write A Book.mid",
        "I Begin To See The Light.mid",
        "I Don't Know What Time.mid",
        "It Don't Mean A Thing (1).mid",
        "If I Were A Bell.mid",
        "I Got Rhythm.mid",
        "I'll Take Romance.mid",
        "I Mean You.mid",
        "Impressions.mid",
        "In A Mellow Tone.mid",
        "In Walked Bud.mid",
        "In Your Own Sweet Way.mid",
        "In A Sentimental Mood.mid",
        "Invitation.mid",
        "I Remember Clifford.mid",
        "Isotope.mid",
        "Israel.mid",
        "It Don't Mean A Thing (2).mid",
        "It's Alright With Me.mid",
        "Jaco.mid",
        "James.mid",
        "Jersey Bounce.mid",
        "Jordu.mid",
        "Joshua.mid",
        "Joy Spring.mid",
        "Just Friends.mid",
        "Just One Of Those Things.mid",
        "Killer Joe.mid",
        "La Fiesta.mid",
        "Ladybird.mid",
        "Lazy Bird.mid",
        "Liberty City.mid",
        "Little Sunflower.mid",
        "Lullaby Of Birdland.mid",
        "Lush Life.mid",
        "Mack The Knife (1).mid",
        "Mack The Knife (2).mid",
        "Manteca.mid",
        "Mayden Voyage.mid",
        "Me And Miss Jones.mid",
        "Misty.mid",
        "Moanin.mid",
        "Monks Mood.mid",
        "Move.mid",
        "Mr PC.mid",
        "My One And Only Love (1).mid",
        "My Foolish Heart.mid",
        "My Funny Valentine.mid",
        "My Love C.mid",
        "My One And Only Love (2).mid",
        "Naima.mid",
        "Nardis.mid",
        "Nature Boy.mid",
        "Nica's Dream.mid",
        "Night And Day.mid",
        "Night Birds.mid",
        "A Night In Tunisia (3).mid",
        "Nostalgia In Times Square.mid",
        "Now's The Time (1).mid",
        "Now's The Time (2).mid",
        "Nutville.mid",
        "Old Devil Moon.mid",
        "On Green Dolphin Street.mid",
        "One Borbon One Scotch.mid",
        "One Note Samba.mid",
        "Opus De Funk.mid",
        "Ornithology.mid",
        "Perdido.mid",
        "Peris Scope.mid",
        "Petite Fleur.mid",
        "Phase Dance.mid",
        "Poinciana.mid",
        "Polka Dots And Moonbeams.mid",
        "Recordame.mid",
        "Reincarnation Of A Lovebird.mid",
        "Rhythm-A-Ning.mid",
        "Romantic Warrior.mid",
        "Round Midnight.mid",
        "Route 66.mid",
        "Ruby My Dear.mid",
        "Salt Peanuts.mid",
        "Satin Doll.mid",
        "Scrapple.mid",
        "Sentimental.mid",
        "Seven Steps To Heaven.mid",
        "Shiny Stockings.mid",
        "Sidewalk.mid",
        "Sidewinder.mid",
        "Since I Fell For You.mid",
        "So Danco Samba.mid",
        "So What (1).mid",
        "So What (2).mid",
        "Soft As A Morning Sunrise.mid",
        "Solar.mid",
        "Song For My Father.mid",
        "Sophisticated Lady.mid",
        "Soul Chicken (1).mid",
        "Soul Chicken (2).mid",
        "Spain.mid",
        "Speak Like A Child.mid",
        "Speak Low (1).mid",
        "Speak Low (2).mid",
        "Speak No Evil.mid",
        "St Thomas.mid",
        "Steamroller Blues.mid",
        "Stella By Starlight.mid",
        "Stolen Moments.mid",
        "Take Five.mid",
        "Take The A Train.mid",
        "Teentown.mid",
        "Tenderly.mid",
        "The Chicken.mid",
        "Time After Time.mid",
        "Touch Of The Blues F.mid",
        "Triste.mid",
        "Tristeza.mid",
        "Very Early.mid",
        "Walking The Dog.mid",
        "Waltz For Debby.mid",
        "Watch What Happens.mid",
        "Watermelon Man (1).mid",
        "Watermelon Man (2).mid",
        "Wave.mid",
        "Well You Needn't.mid",
        "What's New.mid",
        "Whisper Not.mid",
        "Who Can I Turn To.mid",
        "Woody'N You.mid",
        "Yardbird Suite.mid",
        "You And The Night And The Music.mid",
        "You Stepped Out Of A Dream.mid"
    ]

    for i, file in enumerate(sorted(files)):
        sim = int(similar(os.path.basename(file).lower(), renamed[i].lower()) * 100) / 100
        g = renamed[i]

        out_file = os.path.join(
            os.path.dirname(file).replace('JazzStandards2', 'JazzStandards').replace('Raw Data', 'Renamed Raw Data'),
            g
        )

        print(sim, os.path.basename(file), os.path.basename(out_file))

        shutil.copyfile(file, out_file)


phase_2()


