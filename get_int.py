'''
Determine enemy INT, Magic Defense, and Magic Damage Taken based on Ninjutsu spell damage.

ML 37 Hume NIN/WAR without any equipemtn has 134 INT, 28 Magic Attack, 40 Magic Damage, and 506 Ninjutsu skill.
    Including +20 Magic Attack from Category 2 merits, a Lv0 Huge Hornet with INT=6 will take [313, 717, 1147] damage from [Ichi, Ni, San] nukes.
'''
import pandas as pd
import numpy as np
from math import floor
import sys
import argparse
from numba import njit


@njit
def get_mv_ninjutsu(tier, dINT):
    if tier.lower() == "ichi":
        if dINT <= -9:
            m=0.00; v=11.0
        elif dINT <= -1:
            m=0.50; v=16.0
        elif dINT <= 24:
            m=1.00; v=16.0
        elif dINT <= 74:
            m=0.50; v=28.5
        else:
            m=0.00; v=66.0

    elif tier.lower() == "ni":
        if dINT <= -43:
            m=0.00; v=47.0
        elif dINT <= -1:
            m=0.50; v=69.0
        elif dINT <= 112:
            m=1.00; v=69.0
        elif dINT <= 338:
            m=0.50; v=125.5
        else:
            m=0.00; v=295.0

    elif tier.lower() == "san":
        if dINT <= -53:
            m=0.00; v=81.0
        elif dINT <= 1:
            m=1.00; v=134.0
        elif dINT <= 353:
            m=1.50; v=134.0
        else:
            m=0.00; v=665.0

    return(m,v)

@njit
def get_skill_multiplier(tier, skill):

    if tier.lower() == "ichi":
        if skill > 250:
            mult = 2.00
        else:
            mult = 1 + ((skill - 50)/2)/100

    elif tier.lower() == "ni":
        if skill <= 125:
            multi = 1
        elif skill >= 350:
            mult = 2.12
        else:
            mult = 1 + ((skill - 125)/2)/100

    elif tier.lower() == "san":
        if skill <= 275:
            multi = 1
        elif skill >= 500:
            mult = 2.12
        else:
            mult = 1 + ((skill - 275)/2)/100
    
    return mult
    
@njit
def get_damage(dint, player_matk, player_mdmg, player_skill, relicfeet=0, enemy_mdef=0, mdt=0):
    #
    #
    #
    matk_multiplier = (100 + player_matk) / (100 + enemy_mdef)

    damage = []
    for tier in ["ichi", "ni", "san"]:
        skill_multiplier = get_skill_multiplier(tier, player_skill)
        m, v = get_mv_ninjutsu(tier, dint)

        dmg = floor(player_mdmg + v + dint*m)
        dmg = floor(dmg * skill_multiplier)
        dmg = floor(dmg * matk_multiplier)
        dmg = floor(dmg * (1 - mdt/100))
        dmg = floor(dmg * (1.0 + 0.05*relicfeet))

        damage.append(dmg)

    return(damage)

if __name__ == "__main__":
    #
    #
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("ichi", help="Observed Ichi damage.")
    parser.add_argument("ni", help="Observed Ni damage.")
    parser.add_argument("san", help="Observed San damage.")
    parser.add_argument("-int", default="134", help="INT stat. Defaults to 134 (ML37 NIN/WAR Hume)")
    parser.add_argument("-matk", default="28", help="Magic attack from gear, traits, and job points/gitfs. Defaults to 28 from job points and traits. Category1 merits provide +2 for a specific element per merit. Relic+3 head provides +21 from its \"Ninjutsu Damage +21\" stat.")
    parser.add_argument("-mdmg", default="40", help="Magic Damage stat. Defaults to 40.\nNinja gets +40 from job points.")
    parser.add_argument("-skill", default="506", help="Ninjutsu Skill stat.Defaults to 506\n")
    parser.add_argument("-mdef", default="0", help="Enemy magic defense. Defaults to 0.")
    parser.add_argument("-mdt", default="0", help="Enemy magic damage taken. Defaults to 0.")
    parser.add_argument("-merits", default="5", help="Number of merits in Magic Attack Category2. Category2 merits provide +4 per merit (+20 maxed)")
    parser.add_argument("-relicfeet", default="0", help="Relic +3 provide +5% damage (+25% max) and +5 Magic Attack (+25 max) per merit")
    parser.parse_args()

    args = parser.parse_args()

    # Given stats and observed damage, calculate all possible values of damage dealt based on all possible enemy_int.
    # Return the one dINT value (the one enemy_int value) that matches observed damage.
    dint_range = np.arange(-100, 401, 1)

    enemy_mdef = float(args.mdef)
    enemy_mdt = float(args.mdt)

    match = False
    print(f"Checking enemy_mdef = {enemy_mdef:.0f} with enemy_mdt between 0.0 and 50.0")
    while not match:

        # df = DataFrame containing a table of the amount of damage you WOULD do if you used an Ichi/Ni/San nuke with your stats against an enemy with some MDEF and MDT. This is initialized as all 0s.
        df = pd.DataFrame({"dint":dint_range, "ichi":np.zeros_like(dint_range), "ni":np.zeros_like(dint_range), "san":np.zeros_like(dint_range)})

        # Build the DataFrame, filling in theoretical damage values one row at a time.
        for i, row in df.iterrows():
            damage = get_damage(row["dint"], int(args.matk) + 4*int(args.merits)  + 5*int(args.merits)*int(args.relicfeet), int(args.mdmg), int(args.skill), int(args.relicfeet), enemy_mdef, enemy_mdt)
            row["ichi"] = damage[0]
            row["ni"] = damage[1]
            row["san"] = damage[2]

        # Convert the dINT column (player_int - enemy_int) into an "enemy_int" value by subtracting player_int and multiplying by -1.
        # Remove rows where the enemy_int value would be less than 1.
        df["dint"] -= int(args.int)
        df["dint"] *= -1
        df = df[df["dint"] > 0]
        df.rename(columns={"dint":"enemy_int"}, inplace=True)
        # print(df[["enemy_int","ichi","ni","san"]].to_string(index=False))


        # Calculate the difference between the theoretical damage and the observed (input) damage.
        # Create a new column which combines the differences in each spell tier into a single value, using Euclidean distance.
        df["ichi"] -= int(args.ichi)
        df["ni"] -= int(args.ni)
        df["san"] -= int(args.san)
        df["total"] = (df.ichi**2 + df.ni**2 + df.san**2)**0.5

        # Filter the DataFrame into only rows with minimum distance. Ideally is this a single row with distance=0, but it could be multiple rows.
        df0 = df[df.total == min(df.total)]
        enemy_int = df0.enemy_int.values
        if min(df.total) == 0 and len(enemy_int)==1:
            print()
            output_text = f"Perfect match found! Enemy INT = {enemy_int[0]}  Output = [{df0.ichi.values[0] + int(args.ichi)} {df0.ni.values[0] + int(args.ni)} {df0.san.values[0] + int(args.san)}] MDEF={enemy_mdef:4.1f} MDT={enemy_mdt:4.1f}"
            print(output_text)
            match = True
            break

        # No perfect match was found. Increase MDT by 0.1 and retry.
        # If MDT>50, then reset it to 0 and increase MDEF by 1.
        else:
            match = False
            enemy_mdt += 0.1
            if enemy_mdt > 50:
                enemy_mdt = 0
                enemy_mdef += 1
                if enemy_mdef > 50:
                    print("No matches found. Check your input parameters and try again.")
                    break
                print(f"Checking enemy_mdef = {enemy_mdef:.0f} with enemy_mdt between 0.0 and 50.0")
    
            if min(df.total) < 3**0.5 and len(enemy_int)==1:
                print()
                output_text = f"Close match found. Enemy INT = {enemy_int[0]}  Output = [{df0.ichi.values[0] + int(args.ichi)} {df0.ni.values[0] + int(args.ni)} {df0.san.values[0] + int(args.san)}] MDEF={enemy_mdef:4.1f} MDT={enemy_mdt:4.1f}"
                print(output_text)

    print()
    print(f"Input = [{args.ichi} {args.ni} {args.san}]")
    print()
