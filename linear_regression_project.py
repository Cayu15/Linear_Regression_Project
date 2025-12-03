import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import time
import threading
from sklearn.datasets import make_regression
import os
import csv

# Hipotézisfüggvény
def h(theta0, theta1, x):
    return theta0 + theta1 * x

# Költségfüggvény
def cost(theta0, theta1, X, Y):
    m = len(X)
    return np.sum((h(theta0, theta1, X) - Y)**2) / (2*m)

# Gradiens vektor
def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0
    for (xi, yi) in zip(X, y):
        dtheta0 += h(theta0, theta1, xi) - yi
        dtheta1 += (h(theta0, theta1, xi) - yi)*xi
    dtheta0 /= len(X)
    dtheta1 /= len(X)
    return dtheta0, dtheta1

# Paraméterek frissítése
def updateParameters(theta0, theta1, X, Y, alpha):
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, Y)
    theta0 -= alpha * dtheta0
    theta1 -= alpha * dtheta1
    return theta0, theta1

# Lineáris regresszió
def LinearRegression(X, Y, iterations, alpha=0.001, return_steps=False):
    theta0 = np.random.rand() * 10
    theta1 = np.random.rand() * 5
    initial_t0, initial_t1 = theta0, theta1
    cost_ = np.zeros(iterations)
    steps = []

    for i in range(iterations):
        theta0, theta1 = updateParameters(theta0, theta1, X, Y, alpha)
        cost_[i] = cost(theta0, theta1, X, Y)
        if return_steps:
            steps.append((theta0, theta1))
    if return_steps:
        return theta0, theta1, cost_, steps, initial_t0, initial_t1
    else:
        return theta0, theta1, cost_, initial_t0, initial_t1
    
# Adatok
x = np.array([])
y = np.array([])
alpha = 0.01
iterations = 500
steps = []
current_step = 0
running = False
noise = 20
bias = 50

# Tkinter
WIDTH, HEIGHT = 700, 500
MARGIN = 60

root = tk.Tk()
root.title("Lineáris regresszió demonstrálása")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
canvas.grid(row=0, column=0)

info_frame = tk.Frame(root)
info_frame.grid(row=0, column=1, sticky="n", padx=10)

points_text = tk.Text(info_frame, width=25, height=15)
points_text.pack(pady=(0, 10))

stats_text = tk.Text(info_frame, width=25, height=5, bg="#f0f0f0")
stats_text.pack()

# Log fájl
LOG_FILE = "log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        f.write("NumPoints;StartTheta0;StartTheta1;FinalTheta0;FinalTheta1;MSE;Alpha;Iterations;Noise;Bias\n")

def log_run_summary(num_points, start_t0, start_t1, final_t0, final_t1, final_mse,
                    alpha_val=None, iterations_val=None, noise_val=None, bias_val=None):
    def format_val(v):
        if v is None:
            return ''
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, (float, np.floating)):
            if v.is_integer():
                return str(int(v))
            return f"{v:.4f}"
        return str(v)
    
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        f.write(f"{num_points};{format_val(start_t0)};{format_val(start_t1)};"
                f"{format_val(final_t0)};{format_val(final_t1)};{format_val(final_mse)};"
                f"{format_val(alpha_val)};{format_val(iterations_val)};{format_val(noise_val)};{format_val(bias_val)}\n")

# Rajzoló függvények
def draw_axis():
    canvas.delete("axis")
    if len(x) == 0:
        return
    canvas.create_line(MARGIN, HEIGHT-MARGIN, WIDTH-MARGIN, HEIGHT-MARGIN, width=2, tag="axis")
    canvas.create_line(MARGIN, MARGIN, MARGIN, HEIGHT-MARGIN, width=2, tag="axis")
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    for xi in np.linspace(min_x, max_x, num=6):
        cx = MARGIN + (xi - min_x) / (max_x - min_x) * (WIDTH - 2*MARGIN)
        canvas.create_line(cx, HEIGHT-MARGIN-5, cx, HEIGHT-MARGIN+5, tag="axis")
        canvas.create_text(cx, HEIGHT-MARGIN+15, text=f"{xi:.1f}", font=("Arial",8), tag="axis")
    for yi in np.linspace(min_y, max_y, num=6):
        cy = HEIGHT - (MARGIN + (yi - min_y) / (max_y - min_y) * (HEIGHT - 2*MARGIN))
        canvas.create_line(MARGIN-5, cy, MARGIN+5, cy, tag="axis")
        canvas.create_text(MARGIN-30, cy, text=f"{yi:.1f}", font=("Arial",8), tag="axis")

def draw_points():
    canvas.delete("point")
    draw_axis()
    if len(x) == 0:
        return
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    for xi, yi in zip(x, y):
        cx = MARGIN + (xi - min_x) / (max_x - min_x) * (WIDTH - 2*MARGIN)
        cy = HEIGHT - (MARGIN + (yi - min_y) / (max_y - min_y) * (HEIGHT - 2*MARGIN))
        canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill='blue', tag="point")

def draw_line(t0, t1):
    canvas.delete("line")
    if len(x) == 0:
        return
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    x1, x2 = min_x, max_x
    y1, y2 = h(t0, t1, x1), h(t0, t1, x2)
    cx1 = MARGIN + (x1 - min_x)/(max_x - min_x)*(WIDTH-2*MARGIN)
    cy1 = HEIGHT - (MARGIN + (y1 - min_y)/(max_y - min_y)*(HEIGHT-2*MARGIN))
    cx2 = MARGIN + (x2 - min_x)/(max_x - min_x)*(WIDTH-2*MARGIN)
    cy2 = HEIGHT - (MARGIN + (y2 - min_y)/(max_y - min_y)*(HEIGHT-2*MARGIN))
    canvas.create_line(cx1, cy1, cx2, cy2, fill='red', width=2, tag="line")
    update_info(t0, t1)

def update_info(t0, t1):
    c = cost(t0, t1, x, y) if len(x)>0 else 0.0

    points_text.delete("1.0", tk.END)
    points_text.insert(tk.END, f"{'x':>6} {'y':>6}\n")
    points_text.insert(tk.END, "-"*15 + "\n")
    for xi, yi in zip(x, y):
        points_text.insert(tk.END, f"{xi:6.2f} {yi:6.2f}\n")

    stats_text.delete("1.0", tk.END)
    stats_text.insert(tk.END,"\n")
    stats_text.insert(tk.END, f"Theta0: {t0:.4f}\n")
    stats_text.insert(tk.END, f"Theta1: {t1:.4f}\n")
    stats_text.insert(tk.END, f"MSE:    {c:.4f}")

# Adatok beolvasása, generálása, megadása
def load_file():
    global x, y
    file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"), ("All files","*.*")])
    if not file_path:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sniffer = csv.Sniffer()
            sample = f.read(1024)
            f.seek(0)
            dialect = sniffer.sniff(sample, delimiters=",;\t ")
            reader = csv.reader(f, dialect)
            next(reader, None) 
            data_list = []
            for row in reader:
                if len(row) < 2:
                    continue
                data_list.append([float(row[0]), float(row[1])])
            data = np.array(data_list)
        if len(data) == 0:
            messagebox.showerror("Hiba", "A CSV fájl nem tartalmaz adatot.")
            return
        x = data[:,0]
        y = data[:,1]
        draw_points()
    except Exception as e:
        messagebox.showerror("Hiba", f"Nem sikerült betölteni a CSV fájlt:\n{e}")

def generate_data():
    global x, y, noise, bias
    n = simpledialog.askinteger("Generálás", "Pontok száma:", minvalue=2, maxvalue=1000)
    if n is None:
        return
    noise = simpledialog.askfloat("Generálás", "Noise értéke:", minvalue=0, maxvalue=1000)
    bias = simpledialog.askfloat("Generálás", "Bias értéke:", minvalue=-1000, maxvalue=1000)
    X, Y = make_regression(n_samples=n, n_features=1, n_targets=1, noise=noise, bias=bias)
    x = X.flatten()
    y = Y
    draw_points()

def manual_input():
    global x, y
    input_win = tk.Toplevel(root)
    input_win.title("Kézi bevitel")
    tk.Label(input_win, text="Add meg a pontokat soronként x,y formátumban:").pack(padx=10, pady=5)
    text = tk.Text(input_win, width=30, height=15)
    text.pack(padx=10, pady=5)
    def submit():
        global x, y
        try:
            lines = text.get("1.0", tk.END).strip().split("\n")
            x_list, y_list = [], []
            for line in lines:
                xi, yi = map(float, line.split(","))
                x_list.append(xi)
                y_list.append(yi)
            if len(x_list) == 0:
                messagebox.showerror("Hiba", "Nem adtál meg pontokat!")
                return
            x = np.array(x_list)
            y = np.array(y_list)
            draw_points()
            input_win.destroy()
        except Exception as e:
            messagebox.showerror("Hiba", f"Hibás formátum!\n{e}")
    tk.Button(input_win, text="OK", command=submit).pack(pady=5)

# Animáció
def prepare_steps():
    global steps
    if len(x) == 0:
        return None
    theta0, theta1, _, steps, initial_t0, initial_t1 = LinearRegression(x, y, iterations, alpha, return_steps=True)
    final_mse = cost(theta0, theta1, x, y)
    log_run_summary(len(x), initial_t0, initial_t1, theta0, theta1, final_mse,
                    alpha_val=alpha, iterations_val=iterations, noise_val=noise, bias_val=bias)
    return steps, (theta0, theta1, final_mse)

def auto_run():
    global running, current_step, steps
    if len(x) == 0:
        messagebox.showinfo("Adj meg pontokat!")
        return
    running = True
    current_step = 0
    res = prepare_steps()
    if res is None:
        messagebox.showinfo("Nincs adat a futtatáshoz!")
        running = False
        return
    steps, summary = res
    def run():
        global current_step, running
        while running and current_step < len(steps):
            t0, t1 = steps[current_step]
            draw_line(t0, t1)
            current_step += 1
            time.sleep(0.05)
        running = False
        if steps and len(x) > 0:
            final_t0, final_t1, final_mse = summary
            messagebox.showinfo("Kész", f"Final theta0={final_t0:.4f}, Final theta1={final_t1:.4f}\nFinal MSE={final_mse:.4f}")
    threading.Thread(target=run).start()

# Képernyő letörlése
def clear_all():
    global x, y, steps, current_step, running
    running = False
    x = np.array([])
    y = np.array([])
    steps = []
    current_step = 0
    canvas.delete("all")
    points_text.delete("1.0", tk.END)
    stats_text.delete("1.0", tk.END)

# Gombok
main_button_frame = tk.Frame(root)
main_button_frame.grid(row=1, column=0, sticky="w", padx=10, pady=10)

data_frame = tk.Frame(main_button_frame)
data_frame.pack(side=tk.LEFT, padx=5)
tk.Button(data_frame, text="Fájl megnyitása", command=load_file).pack(fill="x", pady=2)
tk.Button(data_frame, text="Pontok generálása", command=generate_data).pack(fill="x", pady=2)
tk.Button(data_frame, text="Pontok megadása", command=manual_input).pack(fill="x", pady=2)

step_frame = tk.Frame(main_button_frame)
step_frame.pack(side=tk.LEFT, padx=5)
tk.Button(step_frame, text="Egyenes illesztése", command=auto_run).pack(fill="x", pady=2)
tk.Button(step_frame, text="Törlés", command=clear_all).pack(fill="x", pady=2)

draw_points()
root.mainloop()