import heapq

def format_waktu(menit):
    hh = (menit // 60) % 24
    mm = menit % 60
    return f"{hh:02d}:{mm:02d}"

class Pasien:
    def __init__(self, id_pasien, waktu_kedatangan, tingkat_kegawatan, estimasi_penanganan):
        self.id_pasien = id_pasien
        self.waktu_kedatangan_str = str(waktu_kedatangan)
        
        if ":" in self.waktu_kedatangan_str:
            hh, mm = map(int, self.waktu_kedatangan_str.split(":"))
            self.waktu_kedatangan = hh * 60 + mm
        else:
            self.waktu_kedatangan = int(waktu_kedatangan)
            
        self.tingkat_kegawatan = int(tingkat_kegawatan) # 1=Merah, 2=Kuning, 3=Hijau
        self.estimasi_penanganan = int(estimasi_penanganan)

class State:
    def __init__(self, jadwal, belum_ditangani, waktu_dokter, g_cost, waktu_saat_ini):
        self.jadwal = jadwal
        self.belum_ditangani = belum_ditangani
        self.waktu_dokter = waktu_dokter
        self.g_cost = g_cost
        self.waktu_saat_ini = waktu_saat_ini

    # Admissible Heuristic: Relaxed problem (asumsi dokter tak terbatas)
    def h_cost(self):
        h = 0
        for p in self.belum_ditangani:
            if self.waktu_saat_ini > p.waktu_kedatangan:
                h += (self.waktu_saat_ini - p.waktu_kedatangan)
        # Penalti untuk pelanggaran kegawatan
        for p in self.belum_ditangani:
            h += (p.tingkat_kegawatan * 5)
        return h

    def f_cost(self):
        return self.g_cost + self.h_cost()

    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

class AStarScheduler:
    def __init__(self, daftar_pasien):
        self.daftar_pasien = daftar_pasien

    def jalankan(self):
        if not self.daftar_pasien:
            return []

        initial_state = State([], self.daftar_pasien, 0, 0, 0)
        frontier = []
        heapq.heappush(frontier, initial_state)

        # Max iter limit
        iterasi = 0
        max_iter = 100000

        best_state = None

        while frontier and iterasi < max_iter:
            current_state = heapq.heappop(frontier)
            iterasi += 1

            if not current_state.belum_ditangani:
                best_state = current_state
                break

            # Batasi search space pada pasien yg available
            pasien_tersedia = [p for p in current_state.belum_ditangani if p.waktu_kedatangan <= current_state.waktu_dokter]
            
            if not pasien_tersedia:
                # Cari pasien dengan waktu kedatangan terawal
                waktu_terawal = min(p.waktu_kedatangan for p in current_state.belum_ditangani)
                pasien_tersedia = [p for p in current_state.belum_ditangani if p.waktu_kedatangan == waktu_terawal]
            
            # Pruning: Ambil pasien dengan prioritas kegawatan tertinggi saja
            kegawatan_tertinggi = min(p.tingkat_kegawatan for p in pasien_tersedia)
            pasien_tersedia = [p for p in pasien_tersedia if p.tingkat_kegawatan == kegawatan_tertinggi]
            
            for p in pasien_tersedia:
                i = current_state.belum_ditangani.index(p)
                # Set start time & calc wait time
                waktu_mulai = max(current_state.waktu_dokter, p.waktu_kedatangan)
                waktu_tunggu = waktu_mulai - p.waktu_kedatangan
                
                # g_cost weighting
                bobot_kegawatan = 4 - p.tingkat_kegawatan # 1 bobotnya 3, 3 bobotnya 1
                tambahan_cost = waktu_tunggu * bobot_kegawatan

                waktu_selesai = waktu_mulai + p.estimasi_penanganan
                
                jadwal_baru = current_state.jadwal.copy()
                jadwal_baru.append({
                    "id": p.id_pasien,
                    "kegawatan": p.tingkat_kegawatan,
                    "waktu_datang": p.waktu_kedatangan_str,
                    "waktu_mulai": format_waktu(waktu_mulai),
                    "waktu_selesai": format_waktu(waktu_selesai),
                    "waktu_tunggu": waktu_tunggu
                })
                
                belum_baru = current_state.belum_ditangani.copy()
                belum_baru.pop(i)
                
                new_state = State(
                    jadwal_baru,
                    belum_baru,
                    waktu_selesai,
                    current_state.g_cost + tambahan_cost,
                    waktu_mulai
                )
                heapq.heappush(frontier, new_state)

        return best_state.jadwal if best_state else []
